#include "arithmetic_coder.h"
#include<iostream>
#include<exception>
// constructor for encoder class:
Encoder::Encoder()
    : encoded_data(), upper(UINT32_MAX), lower(0) {}

uint32_t discretize(float p) {
  // we are going to discretize a floating point number p
  // which is assumed to be in (0, 1).
  // We will discretize it to 16 bits only.
  // We don't want to return 0 or UINT16_MAX though,
  // so we truncate to 1,...,UINT16_MAX-1

  return (uint32_t)(p * (UINT16_MAX - 2)) + 1;
}

uint32_t high_precision_midpoint(uint32_t lower, uint32_t upper, uint32_t p) {
  // the following magic is from
  // https://github.com/byronknoll/cmix/blob/master/src/coder/encoder.cpp
  // This is a fancy way to compute lower + (upper - lower) * p in such a way that the
  // intermediate values do not overflow (as long as p is only 16 bits).
  return lower + ((upper - lower) >> 16) * p +
         (((upper - lower) & 0xffff) * p >> 16);
}


void Encoder::reset(void) {
  encoded_data.clear();
  upper = UINT32_MAX;
  lower = 0;
}


std::vector<py::array_t<uint8_t>> batch_encode_bytes(py::array_t<uint8_t> &bits_as_bytes,  py::array_t<float> &predictions) {
  if (bits_as_bytes.ndim()  != predictions.ndim())
    throw std::runtime_error("Number of dimensions must match");

  for(unsigned int i=0; i< bits_as_bytes.ndim() - 1; i++) {
    if (bits_as_bytes.shape(i) != predictions.shape(i))
      throw std::runtime_error("input shapes must match");
  }

  if (bits_as_bytes.shape(bits_as_bytes.ndim()-1)*8 != predictions.shape(predictions.ndim()-1)) {
    throw std::runtime_error("last dimension of bytes should be 1/8 last dimension of predictions");
  }

  int last_dim = bits_as_bytes.shape(bits_as_bytes.ndim()-1);

  std::vector<int> bytes_shape;

  bytes_shape.push_back(-1);
  bytes_shape.push_back(last_dim);
  
  py::array_t<uint8_t> bytes_view = bits_as_bytes.reshape(bytes_shape);//py::make_tuple(-1, last_dim));//shape);//{-1, last_dim});
  
  std::vector<int> pred_shape;
  pred_shape.push_back(-1);
  pred_shape.push_back(last_dim*8);
  py::array_t<float> pred_view = predictions.reshape(pred_shape);//py::make_tuple(-1, last_dim);//shape);//{-1, last_dim});

  unsigned int num_rows = bytes_view.shape(0);

  
  Encoder encoder;

  std::vector<py::array_t<uint8_t>> encoded_arrays;
  auto unchecked_bytes_view = bytes_view.unchecked<2>();
  auto unchecked_pred_view = pred_view.unchecked<2>();

  for(int i=0; i<num_rows; i++) {
    encoder.reset();
    for(int j=0; j<last_dim; j++) {
      for(int k=0; k<8; k++) {
        encoder.encode_bit(
          (unchecked_bytes_view(i,j)>>(7-k)) &1,
          unchecked_pred_view(i,8*j+k)
        );
      }
    }
    encoder.flush();
    encoded_arrays.push_back(encoder.get_encoded_data());
  }


  return encoded_arrays;
}


void Encoder::encode_bytes(py::array_t<uint8_t> &bits_as_bytes, py::array_t<float> &predictions) {
  if (bits_as_bytes.size() < predictions.size()/8) {
    throw std::runtime_error("prediction length exceeds input bit string length!");
  }
  auto unchecked_bits_as_bytes = bits_as_bytes.unchecked<1>();
  auto unchecked_predictions = predictions.unchecked<1>();
  for(uint32_t i=0;i<predictions.size(); i++) {
    encode_bit(
      (unchecked_bits_as_bytes(i/8)>>((7-i)%8)) & 1,
      unchecked_predictions(i));
  }
}

void Encoder::encode_bit(bool bit, float prediction) {
  const uint32_t discretized_prediction = discretize(prediction);
  const uint32_t mid =
      high_precision_midpoint(lower, upper, discretized_prediction);

  if (bit) {
    // we want to interpret prediction as the P[bit = 1], so
    // if bit is 1, then we want mid to be large, so we should set upper to mid.
    upper = mid;
  } else {
    lower = mid + 1;
  }

  flush_common_bytes();
}

void Encoder::flush_common_bytes() {
  // flush out a byte if possible:
  // top byte is shared if upper 8 bites of upper^lower are the same.
  while (((upper ^ lower) & 0xff000000) == 0) {
    // the upper byte is the same, so:
    // extract the upper byte and write it.
    encoded_data.push_back(upper >> 24);
    // shift out the upper byte and fill in the new zeros with ones.
    upper = ((upper << 8) | 0x000000ff);
    lower = (lower << 8);
  }
}

void Encoder::flush() {
  flush_common_bytes();
  // write top byte of the upper bound.
  encoded_data.push_back(upper >> 24);
}

int Encoder::encoded_size() {
  return encoded_data.size();
}

py::array_t<uint8_t> Encoder::get_encoded_data(void) {
  py::array_t<uint8_t> result(encoded_data.size());
  auto buffer_info = result.request();
  uint8_t* ptr = static_cast<uint8_t*>(buffer_info.ptr);
  std::memcpy(ptr, encoded_data.data(), encoded_data.size() * sizeof(uint8_t));
  return result;
}


std::vector<uint8_t> Encoder::get_encoded_data_vector(void) {
  return encoded_data;
}

Decoder::Decoder(py::array_t<uint8_t> _encoded_data)
    : encoded_data(_encoded_data), lower(0), upper(UINT32_MAX), encoded(0), position(0) {
  // read 4 bytes to fill up encoded...
  read_byte();
  read_byte();
  read_byte();
  read_byte();
}

void Decoder::read_byte() {
  // will shift out the top byte of encoded: make sure
  // you don't need it anymore!
  uint32_t byte;
  auto data  = encoded_data.unchecked<1>();
  if (position >= encoded_data.size()) {
    byte = 0;
  } else {
    byte  = (uint8_t)data(position);
    position++;
  }
  encoded = (encoded<<8) | byte;
}
void Decoder::flush_common_bytes() {
  // flush out a byte if possible:
  // top byte is shared if upper 8 bits of upper^lower are the same.
  while (((upper ^ lower) & 0xff000000) == 0) {
    // the upper byte is the same, so:
    // we don't need that byte of the encoded value anymore.
    // shift it out and read in a new byte.
    read_byte();
    // shift out the upper byte and fill in the new zeros with ones.
    upper = ((upper << 8) | 0x000000ff);
    lower = (lower << 8);
  }
}

bool Decoder::decode_bit(float prediction) {
  const uint32_t discretized_prediction = discretize(prediction);
  const uint32_t mid = high_precision_midpoint(lower, upper, discretized_prediction);

  bool bit;
    
  if (encoded <= mid) {
    // we interpreted prediction as P[bit = 1]. So if
    // if P[bit=1], mid is high and we should be likely to have encoded < mid.
    // In other words: we got the prediction correct.

    // edge-case when encoded == mid:
    // since we set upper=mid if we get the prediction correct in the
    // encoder, in the decoder we consider it correct if encoded == mid.
    bit = 1;

    upper = mid;
  } else {
    bit = 0;

    lower = mid + 1;
  }

  flush_common_bytes();

  return bit;

}


std::vector<bool> bytes_to_bits(const std::vector<uint8_t> &v) {
  std::vector<bool> results;
  results.resize(8 * v.size());
  for(uint32_t i=0;i<v.size();i++) {
    for(int p=0; p<8; p++) {
      results[8*i+p] = (v[i]>>p) & 1;
    }
  }
  return results;
}

std::vector<uint8_t> bits_to_bytes(const std::vector<bool> &v) {
  std::vector<uint8_t> results;
  results.resize(v.size()/8);
  for(uint32_t i=0; i<v.size(); i+=8) {
    uint8_t byte = (v[i] + (v[i+1]<<1) +  (v[i+2]<<2) + (v[i+3]<<3) + (v[i+4]<<4) +  (v[i+5]<<5) + (v[i+6]<<6) + (v[i+7]<<7));
    results[i/8] = byte;
  }
  return results;
}


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(arithmetic_coder, m) {
    m.doc() = R"pbdoc(
        Arithmetic coding example
        -----------------------

        .. currentmodule:: arithmetic_coder

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    m.def("batch_encode_bytes", &batch_encode_bytes);
    
    py::class_<Encoder>(m, "Encoder")
      .def(py::init())
      .def("encode_bit", &Encoder::encode_bit)
      .def("encode_bytes", &Encoder::encode_bytes)
      .def("flush", &Encoder::flush)
      .def("get_encoded_data", &Encoder::get_encoded_data)
      .def("encoded_size", &Encoder::encoded_size);
  
    py::class_<Decoder>(m, "Decoder")
      .def(py::init<py::array_t<uint8_t>>())
      .def("decode_bit", &Decoder::decode_bit);
  
    py::class_<std::vector<uint8_t>>(m, "VectorUInt8", py::buffer_protocol())
     .def_buffer([](std::vector<uint8_t> &v) -> py::buffer_info {
          return py::buffer_info(
              v.data(),                               /* Pointer to buffer */
              sizeof(uint8_t),                          /* Size of one scalar */
              py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
              1,                                      /* Number of dimensions */
              { v.size() },                 /* Buffer dimensions */
              { sizeof(uint8_t) }             /* Strides (in bytes) for each index */
          );
      });
  
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}


