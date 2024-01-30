#ifndef ENCODER_H
#define ENCODER_H


#include<cstdint>
#include<vector>
#include<string>
#include<tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Encoder {

  public:
    Encoder();
    void encode_bit(bool bit, float prediction);
    void encode_bytes(py::array_t<uint8_t> &bits_as_bytes, py::array_t<float> &predictions);
    void flush();
    py::array_t<uint8_t> get_encoded_data(void);
    std::vector<uint8_t> get_encoded_data_vector(void);
    int encoded_size(void);
    void reset(void);

  private:
    void flush_common_bytes();
    std::vector<uint8_t> encoded_data;
    uint32_t lower;
    uint32_t upper;
  
};

std::vector<bool> bytes_to_bits(const std::vector<uint8_t> &v);
std::vector<uint8_t> bits_to_bytes(const std::vector<bool> &v);

class Decoder {

  public:
    Decoder(py::array_t<uint8_t> _encoded_data);
    bool decode_bit(float prediction);

  private:

    void read_byte();
    void flush_common_bytes();

    // std::vector<uint8_t> 
    py::array_t<uint8_t> encoded_data;
    uint32_t lower;
    uint32_t upper;
    uint32_t encoded;
    uint32_t position;
 
};

#endif
