# Tomopy-reconstruction
MicroCT reconstruction using TomoPy
# Reconstructing APS-2BM synchrotron CT data using tomopy

Synchrotron CT data info:

    Camera=dimax

    static scan - 1501 projs x 4 scans = 6004

    back-forth scan - 601 projs x 20 scans = 12020

    volume size= 1008 x 2016 x 2016

    HDF file structure : exchange/data,data_dark,data_white

    Scan files: projection, flat-field (data_white), dark-field (data_dark)


Construction options: all subfolders, single folder, single hdf, single volume, only multiposition, only backNforth
