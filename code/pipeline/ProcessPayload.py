import random
import galois
import numpy as np
import pulp as pl
import time
import os
import math
from collections.abc import Iterable
from typing import Dict, Any, List, Tuple
from collections import Counter
import json
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.make_list import make_list
from utils.synthesize_from_distribution import synthesize_from_distribution
from utils.convert_to_binary import convert_to_binary
from utils.messageSliceBasedOnChunkSize import messageSliceBasedOnChunkSize
from utils.reconstruct_numbers_from_chunks import reconstruct_numbers_from_chunks
# from utils.convert_to_binary import convert_to_binary
# from utils.convert_to_binary import convert_to_binary
# from utils.convert_to_binary import convert_to_binary


from implementations.ParityOverwriteByTopWeightsEncode import ParityOverwriteByTopWeightsEncode
from implementations.OptimizedParityFittingWeightsEncodeAndDecode import OptimizedParityFittingWeightsEncodeAndDecode

NandTvaluesToKvalues = {
    63 : {
        1 : 57, 2: 51, 3: 45, 4: 39, 5: 36, 6: 30,  7: 24,  8: 18
    },
    127 : {
        4 : 99, 5: 92, 6: 85, 7: 78, 8: 71, 9: 71, 10: 64, 11: 57, 12: 50
    },
    255 : {
        8 : 191, 9: 187, 10: 179, 11: 171, 12: 163, 13: 155, 14: 147, 15: 139, 16: 131
    }
}

message_parity_size=63
message_size = 30

# message_parity_size=127
# message_size = 99

# message_parity_size=255
# message_size = 131

job = int(os.environ["job"])
path = "/home/vkamineni/Projects/RECC/pipeline_data/quant_and_job_info.json"
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
print('job',job)
assert payload['job_map'][job]['job_index']==job

start_layer           = payload['job_map'][job]['start_layer'] 
start_offset_in_layer = payload['job_map'][job]['start_offset_in_layer'] 
end_layer             = payload['job_map'][job]['end_layer'] 
end_offset_in_layer   = payload['job_map'][job]['end_offset_in_layer']

AllWeightsInJob = []
for layer in range(start_layer, end_layer+1):
    print(layer,payload['quant_info_records'][layer]['layername'])
    start,end = 0,len(payload['quant_info_records'][layer]['tensor_flattened'])
    if(layer==start_layer):
        start=start_offset_in_layer
    if(layer==end_layer):
        end=end_offset_in_layer
    AllWeightsInJob.extend(payload['quant_info_records'][layer]['tensor_flattened'][start:end])

durations = []
distorsion2_lst = []

ResultingWeightsProcessed = []

curser = 0
for pp in range(math.ceil(len(AllWeightsInJob)/63)):
    # values = make_list(63, 256)
    # values = synthesize_from_distribution(63, offset=128)
    # values = synthesize_from_distribution(message_parity_size, offset=128)
    # values = samples_lst[pp]
    values = AllWeightsInJob[curser:curser+message_parity_size]
    values = (np.array(values) + 128).tolist()
    
    message_bits = convert_to_binary(values, bit_size=8)
    # print('values',values)
#     # print()
    chunks = messageSliceBasedOnChunkSize(message_bits, chunk_size=message_parity_size)

    mutated_chunks2 = []
    count=0
    for chunk in chunks:
        
        # print('start chunk process...',count, pp)
        # print('chunk',chunk)

        start_time = time.time()
        # mutated_chunk2 = OptimizedParityFittingWeightsEncodeAndDecode(
        #             chunk,
        #             message_parity_size=message_parity_size,
        #             message_size=message_size,
        #             # warm_start=mutated_chunk["sliced_message_bits"],
        #             solver='cpsat'
        #         )
        mutated_chunk2 = chunk
        mutated_chunks2.append(mutated_chunk2)
        durations.append(time.time()-start_time)

        # print('end chunk process...',count)
        count+=1
    
    # print('opt2')
    reconstructed_chunks2 = reconstruct_numbers_from_chunks(mutated_chunks2)
    mutated_nums2 = [reconstructed_chunks2[i]['original_number'] for i in range(len(reconstructed_chunks2))]
    
    mutated_nums_neg = (np.array(mutated_nums2) - 128).tolist()
    ResultingWeightsProcessed.extend(mutated_nums_neg)

    distorsion2 = sum([abs(mutated_nums2[i]-values[i]) for i in range(len(mutated_nums2))])/len(values)
    distorsion2_lst.append(distorsion2)

    curser = curser + message_parity_size

    if(pp%200==0):
        print(pp, 'distorsion Cumilative:', sum(distorsion2_lst)/len(distorsion2_lst))
        print()

# save ResultingWeightsProcessed in file path of /home/vkamineni/Projects/RECC/pipeline_data/ResultingWeightsProcessed/{job}
out_dir = Path("/home/vkamineni/Projects/RECC/pipeline_data/ResultingWeightsProcessed")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / f"{str(job)}.json"
tmp_path = out_dir / f".{str(job)}.json.tmp"  # atomic write

with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(ResultingWeightsProcessed, f, separators=(",", ":"))

os.replace(tmp_path, out_path)  # atomic rename
print(f"Saved: {out_path}")

    
print('cpsat-dist',sum(distorsion2_lst)/len(distorsion2_lst))
print('cpsat-duration',sum(durations)/len(durations))
print('----------------------------------------------------------------------------------------------------')