#git clone https://github.com/xindongzhang/ELAN.git
mkdir DIV2K && cd DIV2K
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
unzip DIV2K_train_LR_bicubic_X2.zip
unzip DIV2K_train_LR_bicubic_X3.zip
unzip DIV2K_train_LR_bicubic_X4.zip
cd ..
mkdir benchmark && cd benchmark
cp ../proc.py ./proc.py
wget https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip
wget https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip
wget https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip
wget https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip
unzip kfahv87nfe8ax910l85dksyl2q212voc.zip
unzip igsnfieh4lz68l926l8xbklwsnnk8we9.zip
unzip qgctsplb8txrksm9to9x01zfa4m61ngq.zip
mkdir B100 && cd B100
mv ../image_SRF_2 ./image_SRF_2
mv ../image_SRF_3 ./image_SRF_3
mv ../image_SRF_4 ./image_SRF_4
cd ..
rm readme.txt source_selected.xlsx
unzip 65upg43jjd0a4cwsiqgl6o6ixube6klm.zip
mkdir Urban100 && cd Urban100
mv ../image_SRF_2 ./image_SRF_2
mv ../image_SRF_4 ./image_SRF_4
cd ..
python proc.py Set5
python proc.py Set14
python proc.py B100
python proc.py Urban100
cd ..