# ifndef ADMIN_HPP
# define ADMIN_HPP

# include <cmath>
# include <string>
# include <chrono>
# include <vector>
# include <sstream>
# include <complex>
# include <fftw3.h>
# include <fstream>
# include <iostream>
# include <algorithm>

bool str2bool(std::string s);

void import_binary_float(std::string path, float * array, int n);
void export_binary_float(std::string path, float * array, int n);

void import_text_file(std::string path, std::vector<std::string> &elements);

std::string catch_parameter(std::string target, std::string file);

std::vector<std::string> split(std::string s, char delimiter);

std::vector<std::vector<float>> kaiser_weights(float x, float z, int ix0, int iz0, float dx, float dz); 
std::vector<std::vector<float>> gaussian_weights(float x, float z, int ix0, int iz0, float dx, float dz); 

# endif