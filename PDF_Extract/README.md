

![table_extraction_v2](https://user-images.githubusercontent.com/10793386/139559159-cd23c972-8731-48ed-91df-f3f27e9f4d79.jpg)

This code is from the paper: ["PubTables-1M: Towards comprehensive table extraction from unstructured documents"](https://openaccess.thecvf.com/content/CVPR2022/html/Smock_PubTables-1M_Towards_Comprehensive_Table_Extraction_From_Unstructured_Documents_CVPR_2022_paper.html). And you can see the official link [here](https://github.com/microsoft/table-transformer/tree/main). It is a deep learning model based on object detection for extracting tables from PDFs and images. 

## Training and Evaluation Data

You can download from [Microsoft Research Open Data](https://msropendata.com/) and [Hugging Face](https://huggingface.co/datasets/bsmock/pubtables-1m)

The dataset on Microsoft Research Open Data comes in 5 tar.gz files:

- PubTables-1M-Image_Page_Detection_PASCAL_VOC.tar.gz: Training and evaluation data for the detection model
  - ```/images```: 575,305 JPG files; one file for each page image
  - ```/train```: 460,589 XML files containing bounding boxes in PASCAL VOC format
  - ```/test```: 57,125 XML files containing bounding boxes in PASCAL VOC format
  - ```/val```: 57,591 XML files containing bounding boxes in PASCAL VOC format
- PubTables-1M-Image_Page_Words_JSON.tar.gz: Bounding boxes and text content for all of the words in each page image
  - One JSON file per page image (plus some extra unused files)
- PubTables-1M-Image_Table_Structure_PASCAL_VOC.tar.gz: Training and evaluation data for the structure (and functional analysis) model
  - ```/images```: 947,642 JPG files; one file for each page image
  - ```/train```: 758,849 XML files containing bounding boxes in PASCAL VOC format
  - ```/test```: 93,834 XML files containing bounding boxes in PASCAL VOC format
  - ```/val```: 94,959 XML files containing bounding boxes in PASCAL VOC format
- PubTables-1M-Image_Table_Words_JSON.tar.gz: Bounding boxes and text content for all of the words in each cropped table image
  - One JSON file per cropped table image (plus some extra unused files)
- PubTables-1M-PDF_Annotations_JSON.tar.gz: Detailed annotations for all of the tables appearing in the source PubMed PDFs. All annotations are in PDF coordinates.
  - 401,733 JSON files; one file per source PDF



## Model Training

The code trains models for 2 different sets of table extraction tasks:

1. Table Detection
2. Table Structure Recognition + Functional Analysis

For a detailed description of these tasks and the models, please refer to the paper.

To train, you need to ```cd``` to the ```src``` directory and specify: 1. the path to the dataset, 2. the task (detection or structure), and 3. the path to the config file, which contains the hyperparameters for the architecture and training.

To train the detection model:

```
cd PDF_Extract
python main.py --data_type detection --config_file detection_config.json --data_root_dir /path/to/detection_data
```



## Fine-tuning the Model

If model training is interrupted, it can be easily resumed by using the flag ```--model_load_path /path/to/model.pth``` and specifying the path to the saved dictionary file that contains the saved optimizer state.

If you want to restart training by fine-tuning a saved checkpoint, such as ```model_20.pth```, use the flag ```--model_load_path /path/to/model_20.pth``` and the flag ```--load_weights_only``` to indicate that the previous optimizer state is not needed for resuming training.

Whether fine-tuning or training a new model from scratch, you can optionally create a new config file with different training parameters than the default ones we used. Specify the new config file using: ```--config_file /path/to/new_structure_config.json```. Creating a new config file is useful, for example, if you want to use a different learning rate ```lr``` during fine-tuning.

Alternatively, many of the arguments in the config file can be specified as command line arguments using their associated flags. Any argument specified as a command line argument overrides the value of the argument in the config file.



# Just Extract the Image and Tabular Data

If you want to use the model to extract the image and tabular data directly, you need to download the pre-trained models:

<b>Table Detection:</b>

<table>
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Training Data</th>
      <th>Model Card</th>
      <th>File</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: right;">
      <td>DETR R18</td>
      <td>PubTables-1M</td>
      <td><a href="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_detection_detr_r18.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
  </tbody>
</table>


<b>Table Structure Recognition:</b>

<table>
  <thead>
    <tr style="text-align: left;">
      <th>Model</th>
      <th>Training Data</th>
      <th>Model Card</th>
      <th>File</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: left;">
      <td>TATR-v1.0</td>
      <td>PubTables-1M</td>
      <td><a href="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_structure_detr_r18.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
    <tr style="text-align: left;">
      <td>TATR-v1.1-Pub</td>
      <td>PubTables-1M</td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-Pub">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-Pub/resolve/main/TATR-v1.1-Pub-msft.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
    <tr style="text-align: left;">
      <td>TATR-v1.1-Fin</td>
      <td>FinTabNet.c</td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-Fin">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-Fin/resolve/main/TATR-v1.1-Fin-msft.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
    <tr style="text-align: left;">
      <td>TATR-v1.1-All</td>
      <td>PubTables-1M + FinTabNet.c</td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-All">Model Card</a></td>
      <td><a href="https://huggingface.co/bsmock/TATR-v1.1-All/resolve/main/TATR-v1.1-All-msft.pth">Weights</a></td>
      <td>110 MB</td>
    </tr>
  </tbody>
</table>

Then you can run the code following to get image data:

```sh
cd PDF_Extract
python get_image.py
```

And run the code following to get tabular data:

```shell
cd PDF_Extract
python inference.py --image_dir [your_images] --out_dir [your_output] --mode [detect/structure] --structure_config_path [config_path] --detection_config_path [detection_config_path] --structure_model_path [detection_model_path] --structure_model_path[detection_model_path]
```

