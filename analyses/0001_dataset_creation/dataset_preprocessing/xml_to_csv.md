# xml_to_csv.py

8/9/25

## Purpose:

I made the mistake of saving my image labels in PASCAL VOC format (as
xml files). It’s convenient to read data from csv files when creating a
custom dataset class using pytorch’s
[dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
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
    image_path = Path('/Users/cla24mas/Documents/My_Repos/SC_TSL_15092024_Plate_Detect/train/xml_labels')
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

**Breakdown:**  
- Uses
[pandas.DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
to write ‘xml_df’ to the ‘labels.csv’.  
- ‘index=None’ type=bool default is True, write row names.  

### Next step: create custom Plate_Image_Dataset class using pytorch’s [dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)