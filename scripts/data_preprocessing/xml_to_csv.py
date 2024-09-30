
from pathlib import Path
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in path.iterdir(): 
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('.'):
                xml_list.append(root.find('filename').text)
                xml_list.append(int(root.find('size/depth').text))
                xml_list.append(int(root.find('size/width').text))
                xml_list.append(int(root.find('size/height').text))
                xml_list.append(root.find('object/name').text)
            xmins = []
            ymins = []
            xmaxs = [] # could try 2d array for more concise code loops, but maybe this is easier to interpret
            ymaxs = []
            for i, member in enumerate(root.findall('.//object')):
                xmins.insert(i, int(member.find('bndbox/xmin').text)),
                ymins.insert(i, int(member.find('bndbox/ymin').text)),
                xmaxs.insert(i, int(member.find('bndbox/xmax').text)),
                ymaxs.insert(i, int(member.find('bndbox/ymax').text))
            xml_list.append(xmins)
            xml_list.append(ymins)
            xml_list.append(xmaxs)
            xml_list.append(ymaxs)
            print(f"Processed {xml_file} successfully")
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
    print(xml_list)
    if not xml_list:
        print("No data was extracted from XML files. Check the XML file structure and path.")
        return None
    
    
    xml_df = pd.DataFrame(columns=['filename', 'depth' ,'width', 'height', 'class', 'xmins', 'ymins', 'xmaxs', 'ymaxs'])
    for i in range(int(len(xml_list)/9)):
            xml_df.loc[i] = xml_list[i*9:(i*9)+9]

    
    return xml_df

def main():
    image_path =  Path('/Users/cla24mas/Documents/My_Repos/SC_TSL_15092024_Plate_Detect/train/xml_labels/')
    print(f"Searching for XML files in: {image_path}")
    xml_df = xml_to_csv(image_path)
    if xml_df is not None and not xml_df.empty:
        xml_df.to_csv('train/labels.csv', index=None) 
        print('Successfully converted xml to csv.')
        print(f"CSV file contains {len(xml_df)} rows.")
    else:
        print("Failed to create CSV file. No data was extracted from XML files.")

if __name__ == '__main__':
    main()