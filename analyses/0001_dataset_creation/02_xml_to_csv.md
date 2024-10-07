# Storing XML Label in .CSV File

8/9/25

## Purpose: 

I used [LabelImg](https://pypi.org/project/labelImg/) to label the image
‘samples’ with bounding boxes around any visible plates (or parts of
plates) in the sample. Hence for this dataset, there are two classes:
plate, background. Samples with associated labels were separated from
those without (i.e., no plate visible in sample) by this simple [shell
script](delete_unlabelled.sh). This comprises the ‘positives’ dataset,
i.e., each sample has at least one bounding box label.  
*Note:* as of when I used LabelImg, the installation was broken and
required modifications to two files, as detailed
[here](labelImg_code_modifications.md).

I made the mistake of saving my image labels in PASCAL VOC format (as
.xml files). It’s convenient to read data from csv files when creating a
custom dataset class to be handled by pytorch’s
[DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
Hence, I had to store the bounding box vertices’ coordinates and class
label in a csv file.  

### Parsing XML: 

XML is a hierarchical data format, hence is best represented with a tree
comprised of nodes.  
[ElementTree](https://docs.python.org/3/library/xml.etree.elementtree.html)
provides an API for parsing XML data.  

A typical XML file generated from training data labelling in PASCAL VOC
format:

``` {xml}
<annotation>
    <folder>downsized_plate_subset</folder>
    <filename>0_0_0916161819.png</filename>
    <path>/Users/cla24mas/Documents/My_Repos/SC_TSL_15092024_Plate_Detect-/Training_Data/Unlabelled/augmentation/downsized_plate_subset/0_0_0916161819.png</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>799</width>
        <height>600</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>plate</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>702</xmin>
            <ymin>306</ymin>
            <xmax>799</xmax>
            <ymax>600</ymax>
        </bndbox>
    </object>
</annotation>
```

### Program Overview:

#### Imports:

``` python
from pathlib import Path
import glob
import pandas as pd
import xml.etree.ElementTree as ET
```

#### Parsing xml input file:

``` python
def xml_to_csv(path):
    xml_list = []
    for xml_file in path.iterdir(): 
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('.'):
                value = (
                    root.find('filename').text,
                    int(root.find('size/depth').text),
                    int(root.find('size/width').text),
                    int(root.find('size/height').text),
                    root.find('object/name').text,
                    int(member.find('object/bndbox/xmin').text),
                    int(member.find('object/bndbox/ymin').text),
                    int(member.find('object/bndbox/xmax').text),
                    int(member.find('object/bndbox/ymax').text))
                xml_list.append(value)
            print(f"Processed {xml_file} successfully")
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
```

**Breakdown:**  
- using pathlib, but could equally use glob, a pattern that uses
wildcards to specify a set of files (e.g., the bounding box xml
files).  
- ‘Exception as e’ washelpful in debugging this code.  
- ET.parse() stores most of the elements in the xml file in tree nodes
(excluding XML comments, processing instructions, and doc type input
declarations), each with a tag and a dictionary of attributes accessed
by ‘node.tag’, ‘node.attrib’.  
- child nodes can be iterated over.  
- stores the data of interest in ‘xml_list’, which is returned to
main().  
- returns a pandas dataframe.

#### Main: 

``` python
def main():
    image_path = Path('/Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels')
    print(f"Searching for XML files in: {image_path}")
    xml_df = xml_to_csv(image_path)
    if xml_df is not None and not xml_df.empty:
        xml_df.to_csv('labels.csv', index=None)
        print('Successfully converted xml to csv.')
        print(f"CSV file contains {len(xml_df)} rows.")
    else:
        print("Failed to create CSV file. No data was extracted from XML files.")

if __name__ == '__main__':
    main()
```

    Searching for XML files in: /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162518.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162322.xml successfully
    Error processing /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/classes.txt: syntax error: line 1, column 0
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162536.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162126.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162331.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162255.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162331.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916161801.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162443.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162255.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916161931.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_0_0916161752.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162255.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162331.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162255.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162331.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916161913.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916161752.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162611.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161752.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162554.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162536.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162126.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162144.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162518.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916161846.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162015.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162518.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161904.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162109.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162051.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916161752.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162255.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916161801.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162255.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162331.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_0_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916161948.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162500.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162554.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162126.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162554.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916161752.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916161752.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162331.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916161913.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916161801.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_0_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162331.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162255.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162033.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162109.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162349.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162349.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162126.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162554.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162238.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916161948.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161904.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_0_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_0_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162051.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162109.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162518.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161948.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162238.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161752.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162349.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916161913.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161801.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162554.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162144.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162255.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_0_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161913.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_0_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162033.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_0_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162109.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162015.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161904.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916161659.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162304.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162118.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162340.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162322.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916161922.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162135.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_0_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162247.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916162509.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916161726.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162451.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916161819.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162603.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162621.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162153.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916161939.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_0_0916162545.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916161913.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916161837.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916162313.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162527.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916161855.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161801.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_0_0916162229.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_3_0916162425.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_2_0916161752.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_1_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_2_0916161957.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_3_0916162407.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162211.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_1_0916162500.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162006.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162238.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162024.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_3_0916162100.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/0_2_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_1_0916162434.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162144.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162416.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/1_1_0916162358.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/2_3_0916162042.xml successfully
    Processed /Users/cla24mas/Documents/My_Repos/SC_plate_detect/analyses/0001_dataset_creation/xml_labels/3_2_0916162434.xml successfully
    Failed to create CSV file. No data was extracted from XML files.

**Breakdown:**  
- Uses
[pandas.DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
to write ‘xml_df’ to the ‘labels.csv’.  
- ‘index=None’ type=bool default is True, write row names.  

### Next step: create custom Plate_Image_Dataset class to be passed to Pytorch’s [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
