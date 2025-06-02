/*
 * ------------------------------------------
 *
 *  HIGH-PERFORMANCE INTEGRATED MODELLING SYSTEM (HiPIMS)
 *  Luke S. Smith and Qiuhua Liang
 *  luke@smith.ac
 *
 *  School of Civil Engineering & Geosciences
 *  Newcastle University
 *
 * ------------------------------------------
 *  This code is licensed under GPLv3. See LICENCE
 *  for more information.
 * ------------------------------------------
 *  Domain boundary handling class
 * ------------------------------------------
 *
 */
 #include <vector>
 #include <algorithm>
 #include <boost/lexical_cast.hpp>

 #include "CBoundaryMap.h"
 #include "CBoundaryGridded.h"
 #include "CBoundaryStreamingGridded.h"
 #include "../Datasets/CXMLDataset.h"
 #include "../Datasets/CRasterDataset.h"
 #include "../Domain/Cartesian/CDomainCartesian.h"
 #include "../OpenCL/Executors/COCLBuffer.h"
 #include "../OpenCL/Executors/COCLKernel.h"
 #include "../common.h"


using std::max;
using std::min;
using std::vector;

/*
 *  Constructor
 */
CBoundaryStreamingGridded::CBoundaryStreamingGridded(CDomain* pDomain) {
	this->ucValue = model::boundaries::griddedValues::kValueRainIntensity;
	this->buffer = NULL;
	this->pTransform = NULL;
	this->pBufferConfiguration = NULL;
	this->pBufferValues = NULL;
	this->uiTimeseriesLength = 0;

	this->pDomain = pDomain;
}

/*
 *  Destructor
 */
CBoundaryStreamingGridded::~CBoundaryStreamingGridded() {
	delete this->buffer;
	delete this->pTransform;
	delete this->pBufferConfiguration;
	delete this->pBufferValues;
}

/*
 *	Configure this boundary and load in any related files
 */
bool CBoundaryStreamingGridded::setupFromConfig(XMLElement* pElement, std::string sBoundarySourceDir) {
	char *cBoundaryType, *cBoundaryName, *cBoundaryMask, *cBoundaryInterval, *cBoundaryValue;
	double dInterval = 0.0;

	Util::toLowercase(&cBoundaryType, pElement->Attribute("type"));
	Util::toNewString(&cBoundaryName, pElement->Attribute("name"));
	Util::toNewString(&cBoundaryMask, pElement->Attribute("mask"));
	Util::toLowercase(&cBoundaryInterval, pElement->Attribute("interval"));
	Util::toLowercase(&cBoundaryValue, pElement->Attribute("value"));

	// Must have unique name for each boundary (will get autoname by default)
	this->sName = std::string(cBoundaryName);

	// Convert the interval to a number
	if (CXMLDataset::isValidFloat(cBoundaryInterval)) {
		dInterval = boost::lexical_cast<double>(cBoundaryInterval);
	} else {
		model::doError("Gridded boundary interval is not a valid number.", model::errorCodes::kLevelWarning);
		return false;
	}

	// The gridded data represents...?
	if (cBoundaryValue == NULL || strcmp(cBoundaryValue, "rain-intensity") == 0) {
		this->setValue(model::boundaries::griddedValues::kValueRainIntensity);
	} else if (strcmp(cBoundaryValue, "mass-flux") == 0) {
		this->setValue(model::boundaries::griddedValues::kValueMassFlux);
	} else {
		model::doError(
			"Unrecognised value parameter specified for gridded timeseries data. Currently supported are: "
			"rain-intensity, mass-flux.",
			model::errorCodes::kLevelWarning);
	}

	// Allocate memory for the array of gridded inputs
	// this->uiTimeseriesLength = static_cast<unsigned int>(ceil(pManager->getSimulationLength() / dInterval)) + 1;
	this->uiTimeseriesLength = static_cast<unsigned int>(floor(pManager->getSimulationLength() / dInterval)) + 1;
	//this->pTimeseries = new CBoundaryStreamingGriddedEntry*[this->uiTimeseriesLength];
	CBoundaryGridded::SBoundaryGridTransform* pTransform = NULL;

	this->dTimeseriesInterval = dInterval;
	this->dTimeseriesLength = pManager->getSimulationLength();

	// Deal with the gridded files...
	unsigned long ulEntry = 0;
	for (double dTime = 0.0; dTime <= pManager->getSimulationLength(); dTime += dInterval) {
		const char* cMaskName = Util::fromTimestamp(pManager->getRealStart() + static_cast<unsigned long>(dTime), cBoundaryMask);

		std::string sFilename = sBoundarySourceDir + std::string(cMaskName);

		// Check if the file exists...
		if (!Util::fileExists(sFilename.c_str())) {
			model::doError("Gridded boundary raster missing for " + Util::secondsToTime(dTime) + " with filename '" + sFilename + "'",
						   model::errorCodes::kLevelWarning);
			this->dTimeseriesLength = min(this->dTimeseriesLength, dTime);
			continue;
		}

		this->sFilenames.push_back(sFilename);
		++ulEntry;

		// First raster? Need to come up with a transformation...
		if (pTransform == NULL) {
			CRasterDataset* pRaster = new CRasterDataset();
			pRaster->openFileRead(sFilename);
			pTransform = pRaster->createTransformationForDomain(static_cast<CDomainCartesian*>(this->pDomain));
			delete pRaster;
		}
	}

	this->pTransform = pTransform;

	return true;
}

void CBoundaryStreamingGridded::prepareBoundary(COCLDevice* pDevice,
						COCLProgram* pProgram,
						COCLBuffer* pBufferBed,
						COCLBuffer* pBufferManning,
						COCLBuffer* pBufferTime,
						COCLBuffer* pBufferTimeHydrological,
						COCLBuffer* pBufferTimestep) {
	if (this->pTransform == NULL)
		return;

	// Configuration for the boundary and timeseries data
	if (pProgram->getFloatForm() == model::floatPrecision::kSingle) {
		this->singlePrecision = true;

		sConfigurationSP pConfiguration;

		pConfiguration.TimeseriesEntries = this->uiTimeseriesLength;
		pConfiguration.TimeseriesInterval = this->dTimeseriesInterval;
		pConfiguration.Definition = (cl_uint)this->ucValue;
		pConfiguration.GridRows = this->pTransform->uiRows;
		pConfiguration.GridCols = this->pTransform->uiColumns;
		pConfiguration.GridResolution = this->pTransform->dTargetResolution;
		pConfiguration.GridOffsetX = this->pTransform->dOffsetWest;
		pConfiguration.GridOffsetY = this->pTransform->dOffsetSouth;

		this->pBufferConfiguration = new COCLBuffer("Bdy_" + this->sName + "_Conf", pProgram, true, true, sizeof(sConfigurationSP), true);
		std::memcpy(this->pBufferConfiguration->getHostBlock<void*>(), &pConfiguration, sizeof(sConfigurationSP));

		this->pBufferValues = new COCLBuffer("Bdy_" + this->sName + "_Stream", pProgram, true, true,
											 sizeof(cl_float) * this->pTransform->uiColumns * this->pTransform->uiRows, true);
		/*
		unsigned long ulSize;
		unsigned long ulOffset = 0;
		for (unsigned int i = 0; i < this->uiTimeseriesLength; ++i) {
			void* pGridData = this->pTimeseries[i]->getBufferData( model::floatPrecision::kSingle, this->pTransform );
			ulSize = sizeof( cl_float )* this->pTransform->uiColumns * this->pTransform->uiRows;
			ulOffset += ulSize;
			std::memcpy(
				&( ( this->pBufferTimeseries->getHostBlock<cl_uchar*>() )[ulOffset] ),
				pGridData,
				ulSize
			);
			delete[] pGridData;
		}
		*/
	} else {
		this->singlePrecision = false;
		sConfigurationDP pConfiguration;

		pConfiguration.TimeseriesEntries = this->uiTimeseriesLength;
		pConfiguration.TimeseriesInterval = this->dTimeseriesInterval;
		pConfiguration.Definition = (cl_uint)this->ucValue;
		pConfiguration.GridRows = this->pTransform->uiRows;
		pConfiguration.GridCols = this->pTransform->uiColumns;
		pConfiguration.GridResolution = this->pTransform->dSourceResolution;
		pConfiguration.GridOffsetX = this->pTransform->dOffsetWest;
		pConfiguration.GridOffsetY = this->pTransform->dOffsetSouth;

		this->pBufferConfiguration = new COCLBuffer("Bdy_" + this->sName + "_Conf", pProgram, true, true, sizeof(sConfigurationDP), true);
		std::memcpy(this->pBufferConfiguration->getHostBlock<void*>(), &pConfiguration, sizeof(sConfigurationDP));

		this->pBufferValues = new COCLBuffer("Bdy_" + this->sName + "_Series", pProgram, true, true,
											 sizeof(cl_double) * this->pTransform->uiColumns * this->pTransform->uiRows, true);
		/*
		unsigned long ulSize;
		unsigned long ulOffset = 0;
		for (unsigned int i = 0; i < this->uiTimeseriesLength; ++i)
		{
			void* pGridData = this->pTimeseries[i]->getBufferData( model::floatPrecision::kDouble, this->pTransform );
			ulSize = sizeof( cl_double ) * this->pTransform->uiColumns * this->pTransform->uiRows;
			std::memcpy(
				&((this->pBufferTimeseries->getHostBlock<cl_uchar*>())[ulOffset]),
				pGridData,
				ulSize
			);
			ulOffset += ulSize;
			delete[] pGridData;
		}
		*/
	}

	this->pBufferConfiguration->createBuffer();
	this->pBufferConfiguration->queueWriteAll();
	this->pBufferValues->createBuffer();
	this->pBufferValues->queueWriteAll();

	// Prepare kernel and arguments
	this->oclKernel = pProgram->getKernel("bdy_StreamingGridded");
	COCLBuffer* aryArgsBdy[] = {pBufferConfiguration,
								pBufferValues,
								pBufferTime,
								pBufferTimestep,
								pBufferTimeHydrological,
								NULL,  // Cell states
								pBufferBed,
								pBufferManning};
	this->oclKernel->assignArguments(aryArgsBdy);

	// Dimension the kernel
	// TODO: Need a more sensible group size!
	CDomainCartesian* pDomain = static_cast<CDomainCartesian*>(this->pDomain);
	this->oclKernel->setGlobalSize(ceil(pDomain->getCols() / 8) * 8, ceil(pDomain->getRows() / 8) * 8);
	this->oclKernel->setGroupSize(8, 8);
}

// TODO: Only the cell buffer should be passed here...
void CBoundaryStreamingGridded::applyBoundary(COCLBuffer* pBufferCell) {
	this->oclKernel->assignArgument(5, pBufferCell);
	this->oclKernel->scheduleExecution();
}

void CBoundaryStreamingGridded::streamBoundary(double dTime) {

	// ...
	// TODO: Should we handle all the memory buffer writing in here?...
	unsigned int t = min(static_cast<unsigned int>(floor(dTime / dTimeseriesInterval)), this->uiTimeseriesLength);
	//__private cl_ulong ulTimestep = (cl_ulong)floor( dLclTime / pConfig.TimeseriesInterval );
	//if ( ulTimestep >= pConfig.TimeseriesEntries ) ulTimestep = pConfig.TimeseriesEntries;
	if(this->currentSeriesStep > 0x7FFFFFFF) std::cout << "DEBUG SGB bootstrap css: " << this->currentSeriesStep << " t: " << t << " dTime: " << dTime << " dTimeseriesInterval: " << dTimeseriesInterval << std::endl;
	if(this->currentSeriesStep == t) return;
	std::cout << "DEBUG SGB initiate streaming # css: " << this->currentSeriesStep << " t: " << t << " dTime: " << dTime << " dTimeseriesInterval: " << dTimeseriesInterval  << std::endl;
	this->currentSeriesStep = t;

	// Load the raster...
	CRasterDataset* pRaster = new CRasterDataset();
	pRaster->openFileRead(this->sFilenames[t]);
	this->buffer = new CBoundaryStreamingGriddedEntry(dTime, pRaster->createArrayForBoundary(pTransform));
	delete pRaster;

	if (this->singlePrecision) {
		std::cout << "DEBUG SGB SINGLE" << std::endl;
		void* pGridData = this->buffer->getBufferData(model::floatPrecision::kSingle, this->pTransform);
		unsigned long size = sizeof(cl_float) * this->pTransform->uiColumns * this->pTransform->uiRows;
		std::memcpy(&((this->pBufferValues->getHostBlock<cl_uchar*>())[0]), pGridData, size);
		delete[] pGridData;
	} else {
		std::cout << "DEBUG SGB DOUBLE" << std::endl;
		void* pGridData = this->buffer->getBufferData(model::floatPrecision::kDouble, this->pTransform);
		unsigned long size = sizeof(cl_double) * this->pTransform->uiColumns * this->pTransform->uiRows;

		bool debug_check{false};
		for(uint i{0};i < size/sizeof(cl_double);++i) {
			double * ptr { static_cast<double*>(pGridData) };
			if(ptr[i] > 0) {
				std::cout << "DEBUG SGB non-zero data found: " << ptr[i] << std::endl;
				debug_check = true;
				break;
			}
		}
		if(!debug_check) std::cout << "DEBUG SGB non-zero data check FAILED" << std::endl;

		std::memcpy(&((this->pBufferValues->getHostBlock<cl_uchar*>())[0]), pGridData, size);
		delete[] pGridData;
	}
	this->pBufferValues->queueWriteAll();
}

void CBoundaryStreamingGridded::cleanBoundary() {
	// ...
	// TODO: Is this needed? Most stuff is cleaned in the destructor
}

/*
 *	Timeseries grid data has its own management class
 */
CBoundaryStreamingGridded::CBoundaryStreamingGriddedEntry::CBoundaryStreamingGriddedEntry(double dTime, double* dValues) {
	this->dTime = dTime;
	this->dValues = dValues;
}

CBoundaryStreamingGridded::CBoundaryStreamingGriddedEntry::~CBoundaryStreamingGriddedEntry() {
	delete[] this->dValues;
}

void* CBoundaryStreamingGridded::CBoundaryStreamingGriddedEntry::getBufferData(unsigned char ucFloatMode, CBoundaryGridded::SBoundaryGridTransform* pTransform) {
	void* pReturn;

	if (ucFloatMode == model::floatPrecision::kSingle) {
		cl_float* pFloat = new cl_float[pTransform->uiColumns * pTransform->uiRows];
		for (unsigned long i = 0; i < pTransform->uiColumns * pTransform->uiRows; i++)
			pFloat[i] = this->dValues[i];
		pReturn = static_cast<void*>(pFloat);
	} else {
		pReturn = static_cast<void*>(this->dValues);
	}

	return pReturn;
}
