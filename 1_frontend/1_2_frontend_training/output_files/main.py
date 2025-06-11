import argparse
import torch
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from Lip2Phone import Lip2Phone
from FaceToSpeakerNet import FaceToSpeakerNet
from audiovisual_dataset import AudioVisualDataset
from solver import Solver
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import random
import shutil
import io
import torch.optim as optim
import zipfile

file_list = [
{"id": "1PZcPAcvDfoFfhLp2TtsiHgpmqsT_CBAC", "output": "tensors_2_2.zip"},
{"id": "13mPULtSzjuM7tagTmIh9Rfi5Gwt4TgDP", "output": "tensors_2_4.zip"},
{"id": "1U4wgJ543GjNGv7WdiAKXieP5k8c9_u1K", "output": "tensors_2_5.zip"},
{"id": "1VRj6cGQd8C9tgD_7ZA18pzVgR3nXtjeB", "output": "tensors_2_6.zip"},
{"id": "1Y7dn-N12oh0Wb1_tv9szjbbNXt8qqWTe", "output": "tensors_2_7.zip"},
{"id": "1fpBkIhccnlfeTlIW5ZwpilY2N0JD7ZAP", "output": "tensors_2_8.zip"},
{"id": "10ZEdFVu-m8axqCZZ2akJ7iC_bXWk-6Ej", "output": "tensors_3_1.zip"},
{"id": "1MuI8S2b_AnrTbLJDGiBsbKzWgOAQrlog", "output": "tensors_3_3.zip"},
{"id": "1SBCCdADJCE92WHh6TOt5BapvJq7b6_PA", "output": "tensors_1_8.zip"},
{"id": "14vNALfVpvipd4UPtyhZHtu5M1YnPISR4", "output": "tensors_1_7.zip"},
{"id": "1_cGwcGjhDJJguX29PMsLDgBdli5KoBmN", "output": "tensors_1_6.zip"},
{"id": "1ESKa0qZiWfdw1t9Q05ZTbof9VvrJWY94", "output": "tensors_2_3.zip"},
{"id": "1Mxjp5aER_ia6CotUPai9hdx2Bbr2mex4", "output": "tensors_2_1.zip"},
{"id": "1Vc6QfbfE_bGOVlq1hbQdT5latQclJFR0", "output": "tensors_1_5.zip"},
{"id": "1ayeL95q0AUcvYhLDLrgkvuvuYywiVAR-", "output": "tensors_3_8.zip"},
{"id": "1tglVMpEcgxBsOpnV0h6O2BjdQxRoo71R", "output": "tensors_3_2.zip"},
{"id": "1stjXOhWeGsX-fZQTR3otWwRfwBaNVuer", "output": "tensors_3_5.zip"},
{"id": "1rbEdz8YVDsfhXXotRxNyDsmdrLX5zmf0", "output": "tensors_3_7.zip"},
{"id": "1DuThE6XgL8iUrM0Lc2iPCmMszvGwQJq5", "output": "tensors_3_6.zip"},
{"id": "1auElbZg3yZQcSknZ904N1-txTQyxQOqF", "output": "tensors_3_4.zip"},
{"id": "1wuG6cGH7So82AxuVyS6U343ma_70WpFi", "output": "tensors_1_4.zip"},
{"id": "1BmZUUUL_dnn93hgTVNLXgq55hKtpcz6q", "output": "tensors_1_3.zip"},
{"id": "1YyZ4gIQd8CU9yJuRPC6BgaH7HXFjpEzf", "output": "tensors_1_2.zip"},
{"id": "1CFdUtXonpt-xFbyDZoR5S49j-B3qacNQ", "output": "tensors_1_1.zip"},
]

# !gdown "https://drive.google.com/uc?id=1wDPn0eHP0hvZbtFMhL1F-HJ_1YD7BdIS"
# Authenticate using the service account JSON file
def authenticate_with_service_account(json_keyfile):
    credentials = Credentials.from_service_account_file(json_keyfile, scopes=["https://www.googleapis.com/auth/drive"])
    service = build('drive', 'v3', credentials=credentials)
    return service

# Download file by its ID
def download_file(service, file_id, output_path):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(output_path, 'wb') as file:
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

def download(index):
        # Directory to store the extracted files
    extraction_dir = "."
    # os.makedirs(extraction_dir, exist_ok=True)
    json_keyfile = "avss7.json"
    # Authenticate and download
    service = authenticate_with_service_account(json_keyfile)

    for file in file_list[index*3:(index+1)*3]:
        file_id = file["id"]
        output_path = file["output"]

        download_file(service, file_id, output_path)

        # Check if the file is a ZIP file
        if output_path.endswith(".zip"):
            print(f"Extracting {output_path}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_dir)
            
            # Remove the ZIP file after extraction
            print(f"Removing {output_path}...")
            os.remove(output_path)
    
    print(f"Download of {output_path} complete.")
    print(f"All files downloaded and extracted to {extraction_dir}.")


def main(args):
    train_split = 0.8
    train_split = 0.8
    chunk_indixes = list(range(8))
    shuffled_list = chunk_indixes[:]
    random.shuffle(shuffled_list)
    for index in shuffled_list:
        download(index)
        random_seed = 0
        torch.manual_seed(random_seed)

        dataset = AudioVisualDataset(root_dir=r"./tensors")
        # Compute lengths for train-test split
        dataset_size = len(dataset)
        train_size = int(train_split * dataset_size)
        test_size = dataset_size - train_size


        # Split the dataset
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.distributed:
            torch.manual_seed(0)
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        # Model
        model = Lip2Phone()
        model2  = FaceToSpeakerNet()

        if (args.distributed and args.local_rank ==0) or args.distributed == False:
            print("started on " + args.log_name + '\n')
            print(args)
            print("\nTotal number of parameters: {} \n".format(sum(p.numel() for p in model.parameters())))
            # print(model)
        print(torch.cuda.is_available())
        model = model.cuda()
        model2 = model2.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer2 = optim.Adam(model2.transform.parameters(), lr=1e-4)
        # train_sampler, train_generator = get_dataloader(args,'train')
        # _, test_generator = get_dataloader(args, 'test')
        # args.train_sampler=train_sampler
        

        solver = Solver(args=args,
                    model = model,
                    optimizer = optimizer,
                    model2 = model2,
                    optimizer2 = optimizer2,
                    train_data = train_loader,
                    validation_data = test_loader)
        
        solver.train()
        # Path to the directory
        directory = "./tensors"

        # Remove the directory and its contents
        shutil.rmtree(directory)

        print(f"Directory {directory} and its contents have been removed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("slsyn network training")
    
    # Dataloader
    parser.add_argument('--mix_lst_path', type=str)
    parser.add_argument('--visual_direc', type=str)
    parser.add_argument('--audio_direc', type=str)

    # Training    
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size')
    parser.add_argument('--max_length', default=8, type=int,
                        help='max_length of mixture in training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')   
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of maximum epochs')

    # optimizer
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Init learning rate')
    parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')


    # Log and Visulization
    parser.add_argument('--log_name', type=str, default=None,
                        help='the name of the log')
    parser.add_argument('--use_tensorboard', type=int, default=0,
                        help='Whether to use use_tensorboard')
    parser.add_argument('--continue_from', type=str, default='',
                        help='Whether to use use_tensorboard')

    # Distributed training
    parser.add_argument('--opt-level', default='O0', type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--patch_torch_functions', type=str, default=None)

    parser.add_argument('--effec_batch_size', type=int, default=8,
                    help='Effective batch size')
    parser.add_argument('--accu_grad', type=int, default=1,
                    help='Number of gradient accumulation steps')
    args = parser.parse_args()

    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    main(args)
