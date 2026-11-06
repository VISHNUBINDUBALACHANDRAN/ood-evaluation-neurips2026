import argparse
import os
import pandas as pd
import numpy as np
from datasets import concatenate_datasets, Dataset, Features, Sequence, Value, Array2D, Array3D, load_dataset
from transformers import LayoutLMv3Processor
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils.data import parse_data_paths, mkdir

def main():
    parser = argparse.ArgumentParser(description="Process data.")

    # Adding arguments
    parser.add_argument('--data_folder', type=str, required=True, help='path to data')
    parser.add_argument('--output_folder', type=str, help='path of output dir',default='./')
    parser.add_argument('--processor_path', type=str, required=True,help='path to pre trained processor')

    # Parsing arguments
    args = parser.parse_args()

    # Accessing arguments
    print(f"data_folder: {args.data_folder}")
    print(f"output_folder: {args.output_folder}")
    print(f"processor_path: {args.processor_path}")
    

    assert os.path.exists(args.data_folder), f"folder chosen : {args.data_folder} not found"
    
    
    # parse folder images
    all_paths_imagess = parse_data_paths(args.data_folder)
    # labels
    all_labels = [os.path.basename(os.path.dirname(x)) for x in all_paths_imagess]
    
    df = pd.DataFrame({'path_order':all_paths_imagess,
                  'label':all_labels}
                 )
    
    
    csv_file_save = os.path.join(args.output_folder,'train.csv')
    df.to_csv(csv_file_save,index=False)
    
    data_files = {"train": csv_file_save}
    
    datasets = load_dataset('csv', data_files=data_files)
    
    
    data_to_process = datasets["train"]
    
    processor = LayoutLMv3Processor.from_pretrained(args.processor_path, apply_ocr=True)
    
    # we need to define custom features for v3
    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "pixel_values": Array3D(dtype="float64", shape=(3, 224, 224)),
        }
    )
    
    # transform to enlarge image for OCR
    transformation = transforms.Compose([transforms.Resize(3000)])
    
    device: str = "cpu"
    chunks_data = 5
    
    # split data to chunks
    chunks = np.array_split(np.arange(len(data_to_process)), indices_or_sections=chunks_data)
    
    # path save tensor data
    data_tensor_output = os.path.join(args.output_folder, 'tensor_data')
    
    # Process each chunk sequentially (avoids multiprocessing serialization issues)
    for jj, set_c in enumerate(chunks):
        print(f"Processing chunk {jj+1}/{chunks_data} ({len(set_c)} samples)...")
        
        data_to_process_SAMPLE = data_to_process.select(set_c)
        
        # Process each sample
        all_processed = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "bbox": [],
            "pixel_values": [],
        }
        
        for idx in tqdm(range(len(data_to_process_SAMPLE)), desc=f"Chunk {jj+1}"):
            path = data_to_process_SAMPLE[idx]["path_order"]
            try:
                image = Image.open(path).convert("RGB")
                image = transformation(image)
                
                encoded = processor(
                    image,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                )
                
                for key in all_processed:
                    val = encoded[key]
                    # Strip batch dimension (processor returns batch_size=1 for single image)
                    # val can be list, numpy array, or tensor — always index [0]
                    if isinstance(val, np.ndarray):
                        all_processed[key].append(val[0])
                    elif isinstance(val, list):
                        all_processed[key].append(val[0])
                    else:
                        # torch tensor or other indexable
                        try:
                            all_processed[key].append(val[0])
                        except (IndexError, TypeError):
                            all_processed[key].append(val)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        # Create HuggingFace dataset from processed data
        processed_data = Dataset.from_dict(all_processed, features)
        
        # Add label column
        labels = [data_to_process_SAMPLE[i]["label"] for i in range(min(len(data_to_process_SAMPLE), len(processed_data)))]
        processed_data = processed_data.add_column(name="labels", column=labels[:len(processed_data)])
        
        # Set format to torch tensors
        processed_data.set_format(type="torch", device=device)
        
        path_save = f'part_{jj}'
        folder_path_save = os.path.join(data_tensor_output, path_save)
        mkdir(folder_path_save)
        processed_data.save_to_disk(folder_path_save)
        print(f"Saved chunk {jj+1} to {folder_path_save} ({len(processed_data)} samples)")
    
    # Load and merge all chunks
    from utils.data import fdir
    all_data = fdir(data_tensor_output,
                    specific_end=None,
                    specific_include='part',
                    specific_remove='zip')
    
    
    
    
    # sort data parts load collate and merge to single tensor dataset
    nums = [(l,int(o.split('_')[-1])) for l,o in enumerate(all_data)]
    sorted_list = sorted(nums,key=lambda x: x[1])
    sorted_list_path = [all_data[k[0]] for k in sorted_list]
    
    data_to_collate = []
    
    for j in sorted_list_path:
        data_to_collate.append(Dataset.load_from_disk(j))
        
    
    merged_tensors = concatenate_datasets(data_to_collate)
    
    # save tensor data
    name_saved_collated = os.path.join(args.output_folder, 'colated_tensors')
    
    mkdir(name_saved_collated)
    
    
    merged_tensors.save_to_disk(name_saved_collated)


if __name__ == '__main__':
    
    main()
    