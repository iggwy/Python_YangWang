Hello everyone, this is the program I wrote, which includes reading and plotting data from MESSENGER, BEPICOLOMBO, and TIANWEN-1. 

 2025/4/8-------------------------------
 #Download TIANWEN-1 MOMAG magnetic field data.
 I have uploaded the code for downloading TIANWEN-1 MOMAG magnetic field data.
 The TIANWEN-1 MOMAG data requires login and COOKIES. My code first scrapes all the filenames and IDs, then automatically logs in using the username and password to download the corresponding files.
 These data are in PDS4 format, and I will later write the relevant code for reading and plotting the data.
 ----------------------------
# Download TIANWEN-1 MOMAG Magnetic Field Data

This project provides a Python script that allows users to download TIANWEN-1 MOMAG magnetic field data. The data is in PDS4 format, and the script requires login and cookies for access.

## Features:
- Scrape filenames and IDs of TIANWEN-1 MOMAG magnetic field data.
- Automatic login with username and password.
- Download data in PDS4 format using COOKIES for authentication.
