kernel void sampler_test(read_only image1d_t imgIn, write_only image1d_t imgOut, const int sizeIn, sampler_t sampler, global float* coords)
{
    int j = 0;                              // Index to access the output image
    for (int i = - sizeIn/2; i < sizeIn + sizeIn/2 + 1; ++i)
    {
        float coordIn = i / (float)sizeIn;  // Normalised coordinates in the range [-0.5;1.5] and step 0.1
        float color = read_imagef(imgIn, sampler, coordIn).x;

        coords[j] = coordIn;
        write_imagef(imgOut, j, color);     // The accessed color is just written to the output image
        j++;
    }
}
