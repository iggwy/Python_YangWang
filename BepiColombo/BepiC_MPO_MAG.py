
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import csv
import matplotlib.dates as mdates
import numpy as np
import urllib.request
import requests
import pandas as pd

# Define global variables: MESSENGER data path. The default is the folder where the current script resides.
current_path = os.path.abspath(__file__)
# Get the path to the folder where the current script file resides
BepiC_data_folder = os.path.dirname(current_path)  # To customize the path, modify it here
# The data folder structure is：MESSENGER_data_folder +'/MESSENGER_Data/MAG/Science_MAG/NPY/'
def Read_MAG_TABdata(filename,version='1.0',IO='ib',frame='e2k'):
    #Read TAB files locally and convert them to npy files (improved reading speed later)
    import numpy as np
    import PyFileIO as pf
    if version =='1.0':
      pdsdtype = [('UTC_Str', 'object'),
                ('TIME_OBT', 'object'),
                ('PosX', 'float32'),
                ('PosY', 'float32'),
                ('PosZ', 'float32'),
                ('Bx', 'float32'),
                ('By', 'float32'),
                ('Bz', 'float32'),
                ('TempT1', 'float32'),
                ('TempT2', 'float32'),
                ('TempE', 'float32')]
      data = pf.ReadASCIIData(filename, False, dtype=pdsdtype,SplitChar=',')
      n = data.size
      outdtype = [('UTC_Str', 'object'),
                ('PosX', 'float32'),
                ('PosY', 'float32'),
                ('PosZ', 'float32'),
                ('Bx', 'float32'),
                ('By', 'float32'),
                ('Bz', 'float32'),
                ('Time', 'datetime64[ms]')]
      out = np.recarray(n, outdtype)
      out.UTC_Str=data.UTC_Str
      out.PosX=data.PosX/149597870.7
      out.PosY = data.PosY/149597870.7
      out.PosZ=data.PosZ/149597870.7
      out.Bx=data.Bx
      out.By=data.By
      out.Bz=data.Bz
      for i in range(0,n):
        out.Time[i]=datetime.strptime(data.UTC_Str[i], "%Y-%m-%dT%H:%M:%S.%fZ")
      npyfilepath = BepiC_data_folder + '/BepiC_Data/MPO/MAG/version1.0/1s/NYP/'
      if os.path.isdir(npyfilepath):
          np.save(npyfilepath + 'mag_der_sc_' + IO + '_a001_' + frame + '_00000_' + out.UTC_Str[0][0:4] \
                  + out.UTC_Str[0][5:7] + out.UTC_Str[0][8:10] + '.npy', out)
      else:
          os.makedirs(npyfilepath)
          np.save(npyfilepath + 'mag_der_sc_' + IO + '_a001_'+frame+'_00000_'+out.UTC_Str[0][0:4] \
              + out.UTC_Str[0][5:7] + out.UTC_Str[0][8:10] + '.npy',out)
def Read_Mag_npydata(Time_Range=None,version='1.0',IO='ib',frame='e2k'):
    # Read the local npy file. If the local npy file exists at the corresponding time, the NPY file is read. If no NPY file exists, the PDS automatically downloads the NPY file to the local computer
    Datetime_start = datetime(int(Time_Range[0][0:4]), int(Time_Range[0][5:7]), int(Time_Range[0][8:10]))
    Datetime_end = datetime(int(Time_Range[1][0:4]), int(Time_Range[1][5:7]), int(Time_Range[1][8:10]))
    delta = Datetime_end-Datetime_start
    # Initializes the current date to the start date
    current_date = Datetime_start
    # Create an empty list to store all the dates
    all_dates = []
    # Loop from the start date to the end date
    while current_date <= Datetime_end:
        all_dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    # Check whether an npy data folder exists. If not, create an npy data folder
    if version=='1.0':
     npyfilepath = BepiC_data_folder + '/BepiC_Data/MPO/MAG/version1.0/1s/NYP/'
     if not os.path.isdir(npyfilepath):
            os.makedirs(npyfilepath)
    # If the time range spans multiple files, multiple files are connected
    MAG = {}  # Create an empty dictionary that joins data from multiple npy files
    for date in all_dates:
        if version == '1.0':
            npyfilename = 'mag_der_sc_' + IO + '_a001_'+frame+'_00000_'+date[0:4] \
              + date[5:7] + date[8:10] + '.npy'
        npyfull_path = os.path.join(npyfilepath, npyfilename)
        # Check whether the file exists
        if os.path.exists(npyfull_path):
            MAG_npy = np.load(npyfull_path, allow_pickle=True)
            print("npy.File loaded successfully.")
        else:
            print("npy.File does not exist,and will attempt to convert TAB. data to npy. data.")
            if version == '1.0':
                tabfilepath = BepiC_data_folder + '/BepiC_Data/MPO/MAG/version1.0/1s/TAB/'  # 1s分辨率tab数据文件夹
                if not os.path.isdir(tabfilepath):
                    os.makedirs(tabfilepath)
                tabfilename = 'mag_der_sc_' + IO + '_a001_'+frame+'_00000_'+date[0:4] \
              + date[5:7] + date[8:10] + '.TAB'
            tabfull_path = os.path.join(tabfilepath, tabfilename)
            if os.path.exists(tabfull_path):
                print("tab.File loaded successfully and will be transfered to npy.file.")
                Read_MAG_TABdata(tabfull_path, version)
                MAG_npy = np.load(npyfull_path, allow_pickle=True)
            else:
                print("tab.File does not exist,and will be download from ESA.")
                exist = Download_BepiC_MPO_Mag_Data(date[0:4], date[5:7], date[8:10],version)
                if exist=='Yes':
                  Read_MAG_TABdata(tabfull_path,version)
                  MAG_npy = np.load(npyfull_path, allow_pickle=True)
                else:
                    return None
        for name in MAG_npy.dtype.names:
            # If the array name already exists in merged_arrays, the current array is merged with an existing array
            if name in MAG:
                MAG[name] = np.concatenate((MAG[name], MAG_npy[name]))
            # If the array name does not exist in merged_arrays, the current array is added to merged_arrays
            else:
                MAG[name] = MAG_npy[name]
    Time_range_s = datetime.strptime(Time_Range[0], "%Y-%m-%d %H:%M:%S")  # Can be changed here
    Time_range_e = datetime.strptime(Time_Range[1], "%Y-%m-%d %H:%M:%S")
    index = np.where((MAG['Time'] > Time_range_s) & (MAG['Time'] < Time_range_e))
    returndtype = [('UTC_Str', 'object'),
                ('PosX', 'float32'),
                ('PosY', 'float32'),
                ('PosZ', 'float32'),
                ('Bx', 'float32'),
                ('By', 'float32'),
                ('Bz', 'float32'),
                ('Time', 'datetime64[ms]')]
    returndata = np.recarray(len(index[0]), returndtype)
    returndata.UTC_Str = MAG['UTC_Str'][index[0]]
    returndata.PosX = MAG['PosX'][index[0]]
    returndata.PosY = MAG['PosY'][index[0]]
    returndata.PosZ = MAG['PosZ'][index[0]]
    returndata.Bx = MAG['Bx'][index[0]]
    returndata.By = MAG['By'][index[0]]
    returndata.Bz = MAG['Bz'][index[0]]
    returndata.Time = MAG['Time'][index[0]]
    return returndata
# A global variable that holds time ranges and indexes
time_range_index = 0
def MAG_GUI(time_ranges=None,version='1.0',time_list='None',save_mode=None):
    """Usage: Assign a time range list with n*2 to time_ranges.
       For example:time_ranges=[('2023-06-23 22:00:00','2023-06-24 02:00:00')]
                   MAG_GUI(time_ranges)
      If time_list='ESA',The currently publicly available data is used and the time interval is 1 day
      If save_mode='Yes' ，The time list is automatically plotted and saved locally
    """
    if time_list=='ESA':
      time_ranges=read_MPO_MAG_timelist_from_ESA()
    global time_range_index
    current_time_range = time_ranges[time_range_index]

    # Create main window
    root = tk.Tk()
    root.title("Interactive Plot")
    root.geometry("1100x800")
    root.configure(bg='#f0f0f0')

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Helvetica', 12), padding=10, background='#4CAF50', foreground='white')
    style.map('TButton', background=[('active', '#45a049')])
    style.configure('TLabel', font=('Helvetica', 12), background='#f0f0f0')
    style.configure('TFrame', background='#f0f0f0')

    mag_mso = Read_Mag_npydata(current_time_range,version)
    if mag_mso is None:#If the MPO_MAG data cannot be downloaded, the next time range is automatically read
        root.destroy()
        time_range_index = time_range_index+1
        MAG_GUI(time_ranges, save_mode=save_mode, time_list=time_list)
    bt = np.sqrt(mag_mso.Bx ** 2 + mag_mso.By ** 2 + mag_mso.Bz ** 2)

    # Create plot area
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.subplots_adjust(hspace=0)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Initial plot data
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.plot(mag_mso.Time, bt, 'black', linewidth=0.5)
    ax1.set_title("BepiC MPO MAG", fontsize=16)
    ax1.set_ylabel('Bt', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])

    ax2.plot(mag_mso.Time, mag_mso.Bx, linewidth=0.5, label='Bx')
    ax2.plot(mag_mso.Time, mag_mso.By, linewidth=0.5, label='By')
    ax2.plot(mag_mso.Time, mag_mso.Bz, linewidth=0.5, label='Bz')
    ax2.legend()
    ax2.set_xlabel('UT', fontsize=12)
    ax2.set_ylabel('B', fontsize=12)
    ax2.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save_mode=='Auto':
        if version=='1.0':
          pngfilepath=BepiC_data_folder + '/BepiC_Data/MPO/MAG/version1.0/1s/png/'
          if not os.path.isdir(pngfilepath):
              os.makedirs(pngfilepath)
          pngfilename = os.path.join(pngfilepath,time_list+str(time_range_index)+'-'+\
                                        time_ranges[time_range_index][0][0:10]+'.png')
          plt.savefig(pngfilename)
          plt.close()
          root.destroy()
          time_range_index=time_range_index+1
          MAG_GUI(time_ranges, save_mode='Auto', time_list=time_list)

    # Saves a list of vertical horizontal coordinates and vertical objects
    x_coords = []
    lines1 = []
    lines2 = []

    # A flag variable that controls whether a vertical bar can be inserted
    can_insert_line = True

    # A function that handles click events
    def on_click(event):
        nonlocal can_insert_line
        if event.inaxes and can_insert_line:
            click_x = mdates.num2date(event.xdata)
            click_x = np.datetime64(click_x, 'ms')
            nearest_x = min(mag_mso.Time, key=lambda x_point: abs(x_point - click_x))
            line1 = ax1.axvline(x=nearest_x, color='r', linestyle='--')
            line2 = ax2.axvline(x=nearest_x, color='r', linestyle='--')
            x_coords.append(nearest_x)
            lines1.append(line1)
            lines2.append(line2)
            if len(lines1) == 2:
                can_insert_line = False
            canvas.draw()

    # Bind click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Save the function to CSV file
    def save_to_csv():
        global time_range_index
        file_path = "time_coords.csv"
        new_data = [time_range_index + 1] + x_coords
        if os.path.isfile(file_path):
            # Reading CSV file
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                lines = list(reader)
                # Check whether there is a row corresponding to the current time range index
                for i, line in enumerate(lines):
                    if i > 0 and int(line[0]) == time_range_index + 1:
                        # Replace the data in the current row
                        lines[i] = new_data
                        break
                else:
                    # If there is no row corresponding to the current time_range_index, a new row is added
                    lines.append(new_data)
            # Write the CSV file again
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(lines)
        else:
            # If the file does not exist, write the new data directly
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["number", "X Coordinate 1", "X Coordinate 2"])
                writer.writerow(new_data)
        print(f"Coordinates saved to {file_path}")

    # Function to clear vertical lines
    def clear_lines():
        for line1, line2 in zip(lines1, lines2):
            line1.remove()
            line2.remove()
        lines1.clear()
        lines2.clear()
        x_coords.clear()
        canvas.draw()
        nonlocal can_insert_line
        can_insert_line = True

    # Define amplification function
    def zoom_in():
        if x_coords:
            x_min = min(x_coords)
            x_max = max(x_coords)
            time_range = (mag_mso.Time >= x_min) & (mag_mso.Time <= x_max)
            bt_range = bt[time_range]
            bx_range = mag_mso.Bx[time_range]
            by_range = mag_mso.By[time_range]
            bz_range = mag_mso.Bz[time_range]

            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
            ax1.set_ylim(min(bt_range), max(bt_range))
            ax2.set_ylim(min(np.concatenate([bx_range, by_range, bz_range])),
                         max(np.concatenate([bx_range, by_range, bz_range])))
            clear_lines()
            canvas.draw()

    # Definition reduction function
    def zoom_out():
        ax1.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
        ax2.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
        ax1.set_ylim(min(bt), max(bt))
        ax2.set_ylim(min([min(mag_mso.Bx), min(mag_mso.By), min(mag_mso.Bz)]),
                     max([max(mag_mso.Bx), max(mag_mso.By), max(mag_mso.Bz)]))
        nonlocal can_insert_line
        can_insert_line = True
        canvas.draw()

    # Create button frame
    button_frame = ttk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

    # create button
    clear_button = ttk.Button(button_frame, text="Clear Lines", command=clear_lines)
    clear_button.grid(row=0, column=0, padx=10, pady=10)

    save_button = ttk.Button(button_frame, text="Save to CSV", command=save_to_csv)
    save_button.grid(row=0, column=1, padx=10, pady=10)

    zoom_button = ttk.Button(button_frame, text="Zoom In", command=zoom_in)
    zoom_button.grid(row=0, column=2, padx=10, pady=10)

    zoom_out_button = ttk.Button(button_frame, text="Zoom Out", command=zoom_out)
    zoom_out_button.grid(row=0, column=3, padx=10, pady=10)

    # 定义 Last 按钮功能
    def last_plot():
        global time_range_index
        if time_range_index > 0:
            time_range_index -= 1
            root.destroy()
            MAG_GUI(time_ranges)

    # 定义 Next 按钮功能
    def next_plot():
        global time_range_index
        if time_range_index < len(time_ranges) - 1:
            time_range_index += 1
            root.destroy()
            MAG_GUI(time_ranges)

    next_button = ttk.Button(button_frame, text="Next", command=next_plot)
    next_button.grid(row=0, column=4, padx=10, pady=10)
    last_button = ttk.Button(button_frame, text="Last", command=last_plot)
    last_button.grid(row=0, column=5, padx=10, pady=10)
    # 标记信息Label
    marked_label = ttk.Label(button_frame, text="")
    marked_label.grid(row=0, column=6, padx=10, pady=10)
    # 检查是否已标记
    def check_marked():
        global time_range_index
        file_path = "time_coords.csv"
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:  # 检查文件是否存在且不为空
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # 跳过头行
                lines = list(reader)
                marked = False
                for line in lines:
                    if len(line) > 0 and int(line[0]) == time_range_index + 1:
                        marked = True
                        break
                if marked:
                    marked_label.config(text="Marked: Yes")
                else:
                    marked_label.config(text="Marked: No")
        else:
            marked_label.config(text="Marked: No")

    check_marked()
    root.mainloop()
    root.destroy()
def Download_BepiC_MPO_Mag_Data(year, month,day,version='1.0',IO='ib',frame='e2k'):
    # 下载MESSENGER_MAG_DATA
    #https://psaftp.esac.esa.int/BepiColombo/bc_mpo_mag/data_derived/avg_cal_sc/cruise/ib/e2k/mag_der_sc_ib_a001_e2k_00000_20181215.tab
    #获取下载链接
    if version == '1.0':
        URL = 'https://psaftp.esac.esa.int/BepiColombo/bc_mpo_mag/data_derived/avg_cal_sc/cruise/'+IO \
               + '/' + frame +'/'
        tabfilename = 'mag_der_sc_' + IO + '_a001_'+frame+'_00000_'+year + month+ day + '.tab'
        download_dir = BepiC_data_folder + '/BepiC_Data/MPO/MAG/version1.0/1s/TAB/'  # 1s分辨率tab数据文件夹
    #下载并保存到本地
    download_url = URL + tabfilename
    # Check if the file exists
    file_exists = check_file_exists(download_url)
    if file_exists:
        print("MPO-MAG file exists in ESA web")
        urllib.request.urlretrieve(download_url, download_dir + tabfilename)
        return 'Yes'
    else:
        print("MPO-MAG file does not exist in ESA web")
        return 'No'
def check_file_exists(url):
    try:
        # Send a HEAD request to get only the headers
        response = requests.head(url)

        # Check the HTTP response status code
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            return False
        else:
            return False
    except requests.RequestException as e:
        # Handle request exceptions
        print(f"Request exception: {e}")
        return False

def day_of_year(year, month, day):
    # The conversion date is the day of the year
    from datetime import datetime
    date = datetime(year, month, day)
    start_of_year = datetime(year, 1, 1)
    return (date - start_of_year).days + 1

def doy_to_date(year, day_of_year):
    # How many days of the year are converted to dates
    # The first day of the specified year
    start_of_year = datetime(year, 1, 1)
    # Calculate the target date by increasing the number of days
    target_date = start_of_year + timedelta(days=int(day_of_year) - 1)
    # Format the date as YYYY-MM-DD
    return target_date.strftime('%Y-%m-%d')

def read_MPO_MAG_timelist_from_ESA(version='1.0'):
    if version == '1.0':
     npyfilepath = BepiC_data_folder + '/BepiC_Data/MPO/MAG/version1.0/1s/'
     if not os.path.isdir(npyfilepath):
        os.makedirs(npyfilepath)
     npyfilename='collection_data_derived.npy'
     npyfull_path = os.path.join(npyfilepath, npyfilename)
     # 检查文件是否存在
     if os.path.exists(npyfull_path):
        ESA_time_ranges = np.load(npyfull_path, allow_pickle=True)
     else:
        download_url = 'https://psaftp.esac.esa.int/BepiColombo/bc_mpo_mag/data_derived/collection_data_derived.csv'
        csvfilename='collection_data_derived.csv'
        download_dir = BepiC_data_folder + '/BepiC_Data/MPO/MAG/version1.0/1s/'  # 1s分辨率tab数据文件夹
        urllib.request.urlretrieve(download_url, download_dir + csvfilename)
        # Reading CSV file
        csv_ESA = pd.read_csv(download_dir + csvfilename)
        # Initializes the list of counters and eligible subscripts
        ESAlist = []
        for index,row in csv_ESA.iterrows():
            print(row[1][47:49])
            print(row[1][55:58])
            if row[1][47:49]=='ib' and row[1][55:58]=='e2k' and int(row[1][65:69]) >= 2019 :
                print(row[1][65:69]+'-'+row[1][69:71]+'-'+row[1][71:73]+' '+'00:00:00')
                ESAlist.append((row[1][65:69]+'-'+row[1][69:71]+'-'+row[1][71:73]+' '+'00:00:00',\
                                row[1][65:69]+'-'+row[1][69:71]+'-'+row[1][71:73]+' '+'23:59:59'))
        np.save(os.path.join(download_dir, 'collection_data_derived.npy'), ESAlist)
        ESA_time_ranges=np.load(npyfull_path, allow_pickle=True)
     return ESA_time_ranges

time_ranges=[('2022-04-30 00:00:00','2022-05-01 00:00:00'),('2022-05-06 00:00:00','2022-05-07 00:00:00')]
#MAG_GUI(time_list='ESA',save_mode='Auto')
MAG_GUI(time_ranges)
