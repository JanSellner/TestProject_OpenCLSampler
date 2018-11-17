#ifndef _OPENCL_COMMON_H_
#define _OPENCL_COMMON_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>

inline cl::Device chooseDevice(const int idPlatform = -1, const int idDevice = -1)
{
	std::vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);

    if (idPlatform >= 0 && idDevice >= 0)
    {
        cl::Platform usedPlatform = allPlatforms[idPlatform];
        std::vector<cl::Device> allDevices;
        usedPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);

        return allDevices[idDevice];
    }

	std::cout << "Platforms" << std::endl;
	for (int i = 0; i < allPlatforms.size(); i++) {
		const cl::Platform& p = allPlatforms[i];
		std::cout << "\t" << i << ": " << p.getInfo<CL_PLATFORM_NAME>() << " - " << p.getInfo<CL_PLATFORM_VERSION>() << std::endl;
	}

	// Let the user choose the platform
	int platform = 0;
	while (allPlatforms.size() > 0) {
		std::cout << "Choose platform: ";
		std::cin >> platform;

		if (!std::cin || platform < 0 || platform >= allPlatforms.size()) {
			std::cout << "Invalid platform, choose again." << std::endl;
			std::cin.sync();
			std::cin.clear();
		}
		else {
			break;
		}
	}

	cl::Platform usedPlatform = allPlatforms[platform];

	std::vector<cl::Device> allDevices;
	usedPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);

	std::cout << "Devices on platform " << usedPlatform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	for (int i = 0; i < allDevices.size(); i++) {
		const cl::Device& d = allDevices[i];
		std::cout << "\t" << i << ": " << d.getInfo<CL_DEVICE_NAME>() << " - " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
	}
	
	// Let the user choose the device
	int device = 0;
	while (allDevices.size() > 0) {
		std::cout << "Choose device: ";
		std::cin >> device;

		if (!std::cin || device < 0 || device >= allDevices.size()) {
			std::cout << "Invalid device, choose again." << std::endl;
			std::cin.sync();
			std::cin.clear();
		}
		else {
			break;
		}
	}

	return allDevices[device];
}

inline cl::Device selectFirstGPU()
{
    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);

    // Iterate over all platforms and retrieve all devices, select the first found GPU device
    for (const cl::Platform& platform : allPlatforms)
    {
        try
        {
            std::vector<cl::Device> allDevices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);

            if (allDevices.size() == 1)
            {
                std::cout << "Used platform: " << ": " << platform.getInfo<CL_PLATFORM_NAME>() << " - " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
                std::cout << "Used Device" << ": " << allDevices[0].getInfo<CL_DEVICE_NAME>() << " - " << allDevices[0].getInfo<CL_DEVICE_VERSION>() << std::endl;

                return allDevices[0];
            }
        }
        catch (const cl::Error&)
        {}
    }

    throw cl::Error(1337, "No GPU device found");
}

inline std::string getKernelSource(const std::string& filename)
{
    std::fstream kernelCode(filename, std::ios::in);
    return std::string(std::istreambuf_iterator<char>(kernelCode), std::istreambuf_iterator<char>());
}

#endif //_OPENCL_COMMON_H_
