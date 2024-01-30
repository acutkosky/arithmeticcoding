import arithmetic_coder as ac
import numpy as np
from bitarray import bitarray

encoder = ac.Encoder()


bits = [0, 1, False, False, True, True, True, False, True, False, True, False, False,True, True] + [True] * 16
probs = [0.5, 0.6, 0.3, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.1, 0.1, 0.9,0.9] + [0.9]*16
bits = bitarray(bits)
probs = np.array(probs)
for  b,  p in zip(bits,probs):
    encoder.encode_bit(b, p)

encoder.flush()
    

data = encoder.get_encoded_data()
uint8_array = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

bdata  = bitarray()
bdata.frombytes(data.tobytes())
print("bdata: ",bdata)
print("data bytes: ",data.tobytes())
print("len bdata: ",len(bdata))
print("len input: ",len(bits))

print(data)
print(np.array(data))
    
decoder = ac.Decoder(bdata)

for b, p in zip(bits, probs):
    d_b = decoder.decode_bit(p)
    print("original == decoded: ",int(b) == int(d_b))


