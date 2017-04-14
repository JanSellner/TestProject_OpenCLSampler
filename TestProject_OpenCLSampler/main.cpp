#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <iomanip>
#include "OpenCL_Common.h"

std::vector<float> testMode(const cl::Device& device, cl_addressing_mode mode, std::vector<float>& coords)
{
    try
    {
        // The context is responsible for the host-device interaction and manages the interacting objects (program, kernel, queue)
        cl::Context context({ device });

        // Compile the program
        cl::Program program(context, getKernelSource("sampler_test.cl"));
        //program.build("-cl-std=CL2.0 -I . -g -s sampler_test.cl");    //  -g -s filter.cl for debugging with code builder
        program.build("-cl-std=CL2.0 -I .");    //  -g -s filter.cl for debugging with code builder

        std::vector<float> dataIn = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        std::vector<float> dataOut(dataIn.size() * 2 + 1, 0);
        coords.resize(dataOut.size(), 0);

        // Allocate global memory on the device
        cl::Image1D imgIn(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), dataIn.size());
        cl::Image1D imgOut(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), dataOut.size());
        cl::Sampler sampler(context, CL_TRUE, mode, CL_FILTER_NEAREST);
        cl::Buffer bufferCoords(context, CL_MEM_WRITE_ONLY, sizeof(float) * coords.size());

        // Every command is enqueued in this queue and then executed by the runtime on the device
        cl::CommandQueue queue(context, device);

        // Copy the data to the GPU
        std::array<size_t, 3> origin = { 0, 0, 0 };
        std::array<size_t, 3> imgSize = { dataIn.size(), 1, 1 };
        queue.enqueueWriteImage(imgIn, true, origin, imgSize, dataIn.size() * sizeof(float), 0, dataIn.data());

        cl::Kernel kernel(program, "sampler_test");
        kernel.setArg(0, imgIn);
        kernel.setArg(1, imgOut);
        kernel.setArg(2, static_cast<int>(dataIn.size()));
        kernel.setArg(3, sampler);
        kernel.setArg(4, bufferCoords);

        const cl::NDRange globalProducer(1);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalProducer, cl::NullRange);

        queue.finish();

        // Copy data from device to memory
        std::array<size_t, 3> imgSizeOut = { dataOut.size(), 1, 1 };
        queue.enqueueReadImage(imgOut, true, origin, imgSizeOut, dataOut.size() * sizeof(float), 0, dataOut.data());

        queue.enqueueReadBuffer(bufferCoords, CL_TRUE, 0, coords.size() * sizeof(float), coords.data());

        return dataOut;
    }
    catch (const cl::BuildError& buildError)
    {
        std::cout << buildError.what() << " (" << buildError.err() << "), build info:" << std::endl;
        for (const auto& b : buildError.getBuildLog())
        {
            std::cout << b.second << std::endl;
        }
    }
    catch (const cl::Error& error)
    {
        std::cout << error.what() << " (" << error.err() << ")" << std::endl;
    }

    return{};
}

int main()
{
    cl::Device device = chooseDevice();

    std::vector<float> coords;
    std::vector<float> dataMirror = testMode(device, CL_ADDRESS_MIRRORED_REPEAT, coords);
    std::vector<float> dataRepeat = testMode(device, CL_ADDRESS_REPEAT, coords);
    std::vector<float> dataEdge = testMode(device, CL_ADDRESS_CLAMP_TO_EDGE, coords);
    std::vector<float> dataClamp = testMode(device, CL_ADDRESS_CLAMP, coords);

    std::string nameMirror = "MIRRORED_REPEAT";
    std::string nameRepeat = "REPEAT";
    std::string nameEdge = "CLAMP_TO_EDGE";
    std::string nameClamp = "CLAMP";

    std::cout << "  coords | " << nameMirror << " | " + nameRepeat << " | " + nameEdge << " | " + nameClamp << std::endl;
    std::cout << std::string(8, '-') << "-|-" << std::string(nameMirror.size(), '-') << "-|-" << std::string(nameRepeat.size(), '-') << "-|-" << std::string(nameEdge.size(), '-') << "-|-" << std::string(nameClamp.size(), '-') << std::endl;
    for (size_t i = 0; i < coords.size(); ++i)
    {
        std::cout << std::fixed << std::setprecision(5) << std::setw(8) << coords[i] << " | "
                  << std::setprecision(0) << std::setw(nameMirror.size()) << dataMirror[i] << " | "
                  << std::setw(nameRepeat.size()) << dataRepeat[i] << " | "
                  << std::setw(nameEdge.size()) << dataEdge[i] << " | "
                  << std::setw(nameClamp.size()) << dataClamp[i]
                  << std::endl;
    }
}
