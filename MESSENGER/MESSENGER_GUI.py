import pyspedas
import FitKappaDist
from tkinter import ttk
import csv
import urllib.request
import pandas as pd
import spiceypy
import matplotlib.colors as mcolors
import pds4_tools
import requests
import zipfile
import io
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap, LogNorm
import glob

# Define global variables: MESSENGER data path. The default is the folder where the current script resides.
current_path = os.path.abspath(__file__)
# Get the path to the folder where the current script file resides
MESSENGER_data_folder = os.path.dirname(current_path)  # To customize the path, modify it here
# The data folder structure is：MESSENGER_data_folder +'/MESSENGER_Data/MAG/Science_MAG/NPY/'

def Read_MAG_TABdata(filename, mode='01'):
    #Read TAB files locally and convert them to npy files (improved reading speed), default data mode is' 01 '(1s resolution data)
    import numpy as np
    import PyFileIO as pf
    if mode == '0.05':
        pdsdtype = [('Year', 'int32'),
                    ('DOY', 'int32'),
                    ('Hour', 'int32'),
                    ('Min', 'int32'),
                    ('Sec', 'object'),
                    ('MET', 'float32'),
                    ('Xmso', 'float32'),
                    ('Ymso', 'float32'),
                    ('Zmso', 'float32'),
                    ('Bx', 'float32'),
                    ('By', 'float32'),
                    ('Bz', 'float32')]
    else:
        pdsdtype = [('Year', 'int32'),
                    ('DOY', 'int32'),
                    ('Hour', 'int32'),
                    ('Min', 'int32'),
                    ('Sec', 'object'),
                    ('MET', 'float32'),
                    ('Sample_interval', 'int32'),
                    ('Xmso', 'float32'),
                    ('Ymso', 'float32'),
                    ('Zmso', 'float32'),
                    ('Bx', 'float32'),
                    ('By', 'float32'),
                    ('Bz', 'float32'),
                    ('dBx', 'float32'),
                    ('dBy', 'float32'),
                    ('dBz', 'float32')]
    data = pf.ReadASCIIData(filename, False, dtype=pdsdtype)
    n = data.size
    outdtype = [('Year', 'int32'),
                ('DOY', 'int32'),
                ('Hour', 'int32'),
                ('Min', 'int32'),
                ('Sec', 'object'),
                ('MET', 'float32'),
                ('Time', 'datetime64[ms]'),
                ('Xmso', 'float32'),
                ('Ymso', 'float32'),
                ('Zmso', 'float32'),
                ('Bx', 'float32'),
                ('By', 'float32'),
                ('Bz', 'float32')]
    out = np.recarray(n, outdtype)
    out.Year = data.Year
    out.DOY = data.DOY
    out.Hour = data.Hour
    out.Min = data.Min
    out.Sec = data.Sec
    out.MET = data.MET
    for i in range(0, n):
        time_str = doy_to_date(data.Year[i], data.DOY[i]) + ' ' + str(data.Hour[i]).zfill(2) + ':' + str(
            data.Min[i]).zfill(2) + ':' + data.Sec[i]
        out.Time[i] = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    out.Xmso = data.Xmso / 2440.0
    out.Ymso = data.Ymso / 2440.0
    out.Zmso = data.Zmso / 2440.0
    out.Bx = data.Bx
    out.By = data.By
    out.Bz = data.Bz
    # By default, the NYP data folder is automatically created under the current script folder, and the data is saved in the folder. Of course, you can also modify the path yourself.
    # Gets the path to the current script file

    if mode == '0.05':
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Science_MAG/NPY/'
        if os.path.isdir(npyfilepath):
            np.save(npyfilepath + 'MAGMSOSCI' + str(out.Year[0])[2:4] + str(out.DOY[0]).zfill(3)  + '_V08.npy', out)
        else:
            os.makedirs(npyfilepath)
            np.save(npyfilepath + 'MAGMSOSCI' + str(out.Year[0])[2:4] + str(out.DOY[0]).zfill(3)  + '_V08.npy', out)
    else:
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Reduced_MAG/NPY/'
        if os.path.isdir(npyfilepath):
            np.save(npyfilepath + 'MAGMSOSCIAVG' + str(out.Year[0])[2:4] + str(out.DOY[0]).zfill(3)  + '_' + mode + '_V08.npy',
                    out)
        else:
            os.makedirs(npyfilepath)
            np.save(npyfilepath + 'MAGMSOSCIAVG' + str(out.Year[0])[2:4] + str(out.DOY[0]).zfill(3)  + '_' + mode + '_V08.npy',
                    out)

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
def Read_Mag_npydata(Time_Range, mode='01'):
    #读取本地npy文件，若本地存在对应时间npy文件，则进行读取，若不存在则PDS自动下载到本地
    DOY_start = day_of_year(int(Time_Range[0][0:4]), int(Time_Range[0][5:7]), int(Time_Range[0][8:10]))
    DOY_end = day_of_year(int(Time_Range[1][0:4]), int(Time_Range[1][5:7]), int(Time_Range[1][8:10]))
    #检查是否存在npy数据文件夹,若不存在则创立npy数据文件夹
    if mode == '0.05':
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Science_MAG/NPY/'  #0.05s分辨率数据文件夹
        if not os.path.isdir(npyfilepath):
            os.makedirs(npyfilepath)
    else:
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Reduced_MAG/NPY/'  #01、05、10、60s分辨率数据文件夹
        if not os.path.isdir(npyfilepath):
            os.makedirs(npyfilepath)
    #若时间范围跨多个文件则将连接多个文件
    MAG = {}  #创建连接多个npy文件数据的空字典
    for i in range(DOY_start, DOY_end + 1):
        if mode == '0.05':
            npyfilename = 'MAGMSOSCI' + Time_Range[0][2:4] + str(i).zfill(3) + '_V08.npy'
        else:
            npyfilename = 'MAGMSOSCIAVG' + Time_Range[0][2:4] + str(i).zfill(3) + '_' + mode + '_V08.npy'
        npyfull_path = os.path.join(npyfilepath, npyfilename)
        # 检查文件是否存在
        if os.path.exists(npyfull_path):
            MAG_npy = np.load(npyfull_path, allow_pickle=True)
            print("npy.File loaded successfully.")
        else:
            print("npy.File does not exist,and will attempt to convert TAB. data to npy. data.")
            if mode == '0.05':
                tabfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Science_MAG/TAB/'  # 0.05s分辨率数据文件夹
                if not os.path.isdir(tabfilepath):
                    os.makedirs(tabfilepath)
                tabfilename = 'MAGMSOSCI' + Time_Range[0][2:4] + str(i).zfill(3) + '_V08.TAB'
            else:
                tabfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Reduced_MAG/TAB/'  # 01、05、10、60s分辨率数据文件夹
                if not os.path.isdir(tabfilepath):
                    os.makedirs(tabfilepath)
                tabfilename = 'MAGMSOSCIAVG' + Time_Range[0][2:4] + str(i).zfill(3) + '_' + mode + '_V08.TAB'
            tabfull_path = os.path.join(tabfilepath, tabfilename)
            if os.path.exists(tabfull_path):
                print("tab.File loaded successfully and will be transfered to npy.file.")
                Read_MAG_TABdata(tabfull_path, mode)
                MAG_npy = np.load(npyfull_path, allow_pickle=True)
            else:
                print("tab.File does not exist,and will be download from PDS.")
                Download_MESSENGER_Mag_Data(int(Time_Range[0][0:4]), i, mode)
                Read_MAG_TABdata(tabfull_path, mode)
                MAG_npy = np.load(npyfull_path, allow_pickle=True)
        for name in MAG_npy.dtype.names:
            # 如果数组名已经存在于 merged_arrays 中，则将当前数组与已存在的数组合并
            if name in MAG:
                MAG[name] = np.concatenate((MAG[name], MAG_npy[name]))
            # 如果数组名不存在于 merged_arrays 中，则将当前数组添加到 merged_arrays 中
            else:
                MAG[name] = MAG_npy[name]
    Time_range_s = datetime.strptime(Time_Range[0], "%Y-%m-%d %H:%M:%S")#此处可更改
    Time_range_e = datetime.strptime(Time_Range[1], "%Y-%m-%d %H:%M:%S")
    index = np.where((MAG['Time'] > Time_range_s) & (MAG['Time'] < Time_range_e))
    returndtype = [('Year', 'int32'),
                   ('DOY', 'int32'),
                   ('Hour', 'int32'),
                   ('Min', 'int32'),
                   ('Sec', 'object'),
                   ('MET', 'float32'),
                   ('Time', 'datetime64[ms]'),
                   ('Xmso', 'float32'),
                   ('Ymso', 'float32'),
                   ('Zmso', 'float32'),
                   ('Bx', 'float32'),
                   ('By', 'float32'),
                   ('Bz', 'float32'),
                   ('t_tag', 'U')]
    returndata = np.recarray(len(index[0]), returndtype)
    returndata.Year = MAG['Year'][index[0]]
    returndata.DOY = MAG['DOY'][index[0]]
    returndata.Hour = MAG['Hour'][index[0]]
    returndata.Min = MAG['Min'][index[0]]
    returndata.Sec = MAG['Sec'][index[0]]
    returndata.MET = MAG['MET'][index[0]]
    returndata.Time = MAG['Time'][index[0]]
    returndata.Xmso = MAG['Xmso'][index[0]]
    returndata.Ymso = MAG['Ymso'][index[0]]
    returndata.Zmso = MAG['Zmso'][index[0]]
    returndata.Bx = MAG['Bx'][index[0]]
    returndata.By = MAG['By'][index[0]]
    returndata.Bz = MAG['Bz'][index[0]]
    return returndata
# Define global variables to hold time ranges and indexes
time_range_index = 3086#1414#511
def Plot_GUI(time_ranges=None, mode='01', time_list='MP_Sun',Orbit=None):
    # Magnetic field data interface, you can set the time range list (time_list = None at this time)
    # If time_list='MP_Sun', assign Sun's magnetopause event list to time_ranges
    # If time_list='BS_Sun', assign Sun's bow shock event list to time_ranges

    if time_list=='MP_Sun':
        mplist = np.load(MESSENGER_data_folder+'/mplist.npy',allow_pickle=True)
        # Initializes the time ranges list
        time_ranges = []
        type = mplist['type']
        # Traverse the mplist, combining start and end into tuples to add to time_ranges
        for record in mplist:
            start = record['start']
            end = record['end']

            # The start time and end time of parsing are datetime objects
            start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

            # Expand the time range: Add 2 minute each time
            extended_start = start_time - timedelta(minutes=10)
            extended_end = end_time + timedelta(minutes=10)
            if record['MP_pos'][0]>0 : #and record['shear_angle']>120
            # Adds the extended time range to the time ranges
             time_ranges.append((extended_start.strftime('%Y-%m-%d %H:%M:%S'), extended_end.strftime('%Y-%m-%d %H:%M:%S')))

    if time_list=='BS_Sun':
        bslist = np.load(MESSENGER_data_folder+'/bslist.npy',allow_pickle=True)
        # Initializes the time ranges list
        time_ranges = []
        # Iterate over the bslist, combining start and end into tuples to add to time_ranges
        for record in bslist:
            start = record['start']
            end = record['end']
            # 解析开始时间和结束时间为 datetime 对象
            start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

            # 扩展时间范围：前后各增加1分钟
            extended_start = start_time - timedelta(minutes=2)
            extended_end = end_time + timedelta(minutes=2)

            # Adds the extended time range to the time ranges
            time_ranges.append((extended_start.strftime('%Y-%m-%d %H:%M:%S'), extended_end.strftime('%Y-%m-%d %H:%M:%S')))

    global time_range_index
    if time_list==None :
        current_time_range = time_ranges[0]
    else:
        current_time_range = time_ranges[time_range_index]
        index=np.where((mplist['MP_pos'][:,0]>0) ) #& (mplist['shear_angle']>120)
        print(len(index[0]))
        Lf=mplist['angle_Lf'][index]
        shear=mplist['shear_angle'][index]
        angle_Lf=str(Lf[time_range_index])
        angle_shear = str(shear[time_range_index])
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

    mag_mso = Read_Mag_npydata(current_time_range, mode)
    bt = np.sqrt(mag_mso.Bx ** 2 + mag_mso.By ** 2 + mag_mso.Bz ** 2)
    # 获取 FIPS 数据
    FIPS_data = Read_FIPS_npydata(Time_Range=current_time_range)
    if not isinstance(FIPS_data, np.ndarray):
        time_range_index=time_range_index+1
        print(time_range_index)
        root.destroy()
        Plot_GUI(time_ranges, mode)
    # 将时间转换为数字格式
    posix_time = mdates.date2num(FIPS_data['Time'])
    n = len(posix_time)  # n 的值
    posix_time_2d = np.tile(posix_time[:, np.newaxis], (1, 64))
    # 创建自定义的颜色映射，将NaN部分变为白色
    cmap = plt.get_cmap('jet')
    cmap_list = [cmap(i) for i in range(cmap.N)]
    custom_cmap = ListedColormap(cmap_list)
    # 将没有数据的部分设为NaN
    PFlux = FIPS_data['PFlux']#PFlux
    PFlux[PFlux == 0] = np.nan

    # 创建图形和子图
    fig = plt.figure(figsize=(6, 4))
    # 第一个子图
    ax1 = fig.add_axes([0.1, 0.8, 0.8, 0.15])  # [left, bottom, width, height]
    ax1.plot(mag_mso.Time, bt, 'black', linewidth=0.5)
    if time_list=='MP_Sun':
       ax1.set_title("MESSENGER_"+str(time_range_index)+'_'+current_time_range[0][0:10]+'_Lf:'+angle_Lf+' shear:'+angle_shear, fontsize=16)
    else:
        ax1.set_title(
        "MESSENGER_" +  '_' + current_time_range[0][0:10],
            fontsize=16)
    ax1.set_ylabel('Bt', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])

    # 第二个子图
    ax2 = fig.add_axes([0.1, 0.65, 0.8, 0.15])  # [left, bottom, width, height]
    ax2.plot(mag_mso.Time, mag_mso.Bx, linewidth=0.5, label='Bx')
    ax2.plot(mag_mso.Time, mag_mso.By, linewidth=0.5, label='By')
    ax2.plot(mag_mso.Time, mag_mso.Bz, linewidth=0.5, label='Bz')
    ax2.legend()
    ax2.set_ylabel('B', fontsize=12)
    ax2.get_xaxis().set_visible(False)
    ax2.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 第三个子图（伪彩图）
    ax3 = fig.add_axes([0.1, 0.5, 0.8, 0.15])  # [left, bottom, width, height]
    pcm = ax3.pcolormesh(posix_time_2d, FIPS_data['Energy'] * 1000, PFlux, shading='auto', cmap=custom_cmap,
                         norm=LogNorm(vmin=1e1, vmax=1e10))
    ax3.set_ylim(FIPS_data['Energy'].min() * 1000, FIPS_data['Energy'].max() * 1000)
    # 创建新的Axes对象来放置colorbar
    cax = fig.add_axes([0.91, 0.5, 0.02, 0.15])  # [left, bottom, width, height]
    cbar = fig.colorbar(pcm, cax=cax, orientation='vertical', label='PFlux')
    ax3.set_ylabel('Energy (eV)')
    ax3.set_yscale('log')
    ax3.get_xaxis().set_visible(False)
    # 设置横轴格式为小时和分钟


    ax4 = fig.add_axes([0.1, 0.35, 0.8, 0.15])  # [left, bottom, width, height]
    ax4.plot(posix_time_2d, FIPS_data['n_obs'], 'black', linewidth=0.5)#N_proton
    ax4.set_ylabel('N (cm-3)')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()  # 自动调整日期标签的格式
    ax4.set_xlim(mdates.date2num(mag_mso.Time[0]), mdates.date2num(mag_mso.Time[-1]))
    ax4.get_xaxis().set_visible(False)

    ax5 = fig.add_axes([0.1, 0.2, 0.8, 0.15])  # [left, bottom, width, height]
    ax5.plot(posix_time_2d, FIPS_data['v_obs']/1000, 'black', linewidth=0.5)#T_proton
    ax5.set_xlabel('Time')
    ax5.set_ylabel('V km/s')
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()  # 自动调整日期标签的格式
    ax5.set_xlim(mdates.date2num(mag_mso.Time[0]), mdates.date2num(mag_mso.Time[-1]))
    ax5.get_xaxis().set_visible(False)

    ax6 = fig.add_axes([0.1, 0.05, 0.8, 0.15])  # [left, bottom, width, height]
    ax6.plot(posix_time_2d,FIPS_data['n_obs']*FIPS_data['v_obs'], 'black', linewidth=0.5)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Moment')
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()  # 自动调整日期标签的格式
    ax6.set_xlim(mdates.date2num(mag_mso.Time[0]), mdates.date2num(mag_mso.Time[-1]))


    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
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
        new_data = [time_range_index+1] + x_coords + [type[time_range_index]]
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
                writer.writerow(["number", "X Coordinate 1", "X Coordinate 2","type"])
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
            time_range = (FIPS_data['Time'] >= x_min) & (FIPS_data['Time'] <= x_max)
            N_range = FIPS_data['n_obs'][time_range]
            T_range = FIPS_data['v_obs'][time_range]/1000
            K_range = FIPS_data['n_obs'][time_range]*FIPS_data['v_obs'][time_range]

            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
            ax3.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
            ax4.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
            ax5.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
            ax6.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
            ax1.set_ylim(min(bt_range), max(bt_range))
            ax2.set_ylim(min(np.concatenate([bx_range, by_range, bz_range])),
                         max(np.concatenate([bx_range, by_range, bz_range])))
            ax4.set_ylim(min(N_range), max(N_range))
            ax5.set_ylim(min(T_range), max(T_range))
            ax6.set_ylim(min(K_range), max(K_range))
            clear_lines()
            canvas.draw()

    # Definition reduction function
    def zoom_out():
        ax1.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
        ax2.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
        ax3.set_xlim(mdates.date2num(mag_mso.Time[0]), mdates.date2num(mag_mso.Time[-1]))
        ax4.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
        ax5.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
        ax6.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
        ax1.set_ylim(min(bt), max(bt))
        ax2.set_ylim(min([min(mag_mso.Bx), min(mag_mso.By), min(mag_mso.Bz)]),
                     max([max(mag_mso.Bx), max(mag_mso.By), max(mag_mso.Bz)]))
        ax4.set_ylim(min(FIPS_data['n_obs']), max(FIPS_data['n_obs']))
        ax5.set_ylim(min(FIPS_data['v_obs']/1000), max(FIPS_data['v_obs']/1000))
        ax6.set_ylim(min(FIPS_data['n_obs']*FIPS_data['v_obs']), max(FIPS_data['n_obs']*FIPS_data['v_obs']))
        nonlocal can_insert_line
        can_insert_line = True
        canvas.draw()

    # Create button frame
    button_frame = ttk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

    # Create button
    clear_button = ttk.Button(button_frame, text="Clear Lines", command=clear_lines)
    clear_button.grid(row=0, column=0, padx=10, pady=10)

    save_button = ttk.Button(button_frame, text="Save to CSV", command=save_to_csv)
    save_button.grid(row=0, column=1, padx=10, pady=10)

    zoom_button = ttk.Button(button_frame, text="Zoom In", command=zoom_in)
    zoom_button.grid(row=0, column=2, padx=10, pady=10)

    zoom_out_button = ttk.Button(button_frame, text="Zoom Out", command=zoom_out)
    zoom_out_button.grid(row=0, column=3, padx=10, pady=10)

    # Define the Last button function
    def last_plot():
        global time_range_index
        if time_range_index > 0:
            time_range_index -= 1
            print(time_range_index)
            root.destroy()
            Plot_GUI(time_ranges,mode)

    # Define the Next button function
    def next_plot():
        global time_range_index
        if time_range_index < len(time_ranges) - 1:
            time_range_index += 1
            print(time_range_index)
            root.destroy()
            Plot_GUI(time_ranges,mode)

    next_button = ttk.Button(button_frame, text="Next", command=next_plot)
    next_button.grid(row=0, column=4, padx=10, pady=10)
    last_button = ttk.Button(button_frame, text="Last", command=last_plot)
    last_button.grid(row=0, column=5, padx=10, pady=10)
    # Label information
    marked_label = ttk.Label(button_frame, text="")
    marked_label.grid(row=0, column=6, padx=10, pady=10)

    # Check if it is marked
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
def Download_MESSENGER_Mag_Data(year, doy, mode):
    # Download MESSENGER_MAG_DATA,
    Month = int(doy_to_date(year, doy)[5:7])
    # Get the first day of the month
    first_day = datetime(year, Month, 1)
    # Get the first day of the next month
    if Month == 12:
        next_month_first_day = datetime(year + 1, 1, 1)
    else:
        next_month_first_day = datetime(year, Month + 1, 1)
    # Get the last day of the month
    last_day = next_month_first_day - timedelta(days=1)
    # Calculate the first day of the month and the last day of the year
    first_day_of_year = datetime(year, 1, 1)
    first_day_of_year_number = (first_day - first_day_of_year).days + 1
    last_day_of_year_number = (last_day - first_day_of_year).days + 1
    first_last_month_str = str(first_day_of_year_number).zfill(3) + '_' + str(last_day_of_year_number).zfill(3) + '_'
    Month_str = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    # # Get download link
    if mode == '0.05':
        URL = 'https://search-pdsppi.igpp.ucla.edu/ditdos/download?id=pds://PPI/mess-mag-calibrated/data/mso/' + str(
            year) + '/' + first_last_month_str + Month_str[Month - 1] + '/'
        tabfilename = 'MAGMSOSCI' + str(year)[2:4] + str(doy).zfill(3) + '_V08.TAB'
        download_dir = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Science_MAG/TAB/'
    else:
        URL = 'https://search-pdsppi.igpp.ucla.edu/ditdos/download?id=pds://PPI/mess-mag-calibrated/data/mso-avg/' + str(
            year) + '/' + first_last_month_str + Month_str[Month - 1] + '/'
        tabfilename = 'MAGMSOSCIAVG' + str(year)[2:4] + str(doy).zfill(3) + '_' + mode + '_V08.TAB'
        download_dir = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Reduced_MAG/TAB/'
    # Download and save to local
    download_url = URL + tabfilename
    urllib.request.urlretrieve(download_url, download_dir + tabfilename)
def Read_Sun_MP_BS():
    # Reading CSV file
    Sun_csv='G:\SpaceScience\MESSENGER\Boundary_Sun.csv'
    df_Sun = pd.read_csv(Sun_csv)

    # Initializes the list of counters and eligible subscripts
    mp_indices = []
    bs_indices = []

    # Iterate over the DataFrame, recording eligible subscripts
    for i in range(len(df_Sun)):
        Time_range_s = datetime.strptime(df_Sun['start'][i], "%Y-%m-%d %H:%M:%S")  # 此处可更改
        Time_range_e = datetime.strptime(df_Sun['end'][i], "%Y-%m-%d %H:%M:%S")

        if df_Sun['Type'][i][0:2] == 'mp' :
              mp_indices.append(i)
        else:
              bs_indices.append(i)

    # Defines the dtype of the array of records
    mptype = [('start', 'object'),
             ('end', 'object'),
             ('B_msh', 'float32',(3,)),
             ('B_msp', 'float32',(3,)),
             ('MP_pos', 'float32', (3,)),
             ('L_vec', 'float32', (3,)),
             ('angle_Lf', 'float32'),
             ('shear_angle', 'float32'),
             ('MET_diff', 'float32'),
             ('num', 'object'),
             ('type', 'object')]
    bstype = [('start', 'object'),
             ('end', 'object'),
             ('type', 'object')]

    # Creates an empty array of records
    mplist = np.recarray(len(mp_indices), dtype=mptype)
    bslist = np.recarray(len(bs_indices), dtype=bstype)
    # Populate the array of records with subscripts
    for idx, i in enumerate(mp_indices):
         print(i)
         B_msh,B_msp,angle_Lf,shear_angle,MET_diff,MP_pos,L_vec=get_mp_parameter([df_Sun['start'][i],df_Sun['end'][i]],'mva',df_Sun['Type'][i])
         mplist[idx] = (df_Sun['start'][i], df_Sun['end'][i], B_msh,B_msp,MP_pos,L_vec,angle_Lf,shear_angle,MET_diff, df_Sun['num'][i],df_Sun['Type'][i])
         print(df_Sun['num'][i])
    # Save as an npy file
    np.save(os.path.join(MESSENGER_data_folder, 'mplist.npy'), mplist)
    for idx, i in enumerate(bs_indices):
        if i > 5300 : #5309
          print(i)
          bslist[idx] = (df_Sun['start'][i], df_Sun['end'][i], df_Sun['Type'][i])
    np.save(os.path.join(MESSENGER_data_folder, 'bslist.npy'), bslist)
def get_mp_parameter(trange,method=None,type=None):
    if method=='mva':
     start_time = datetime.strptime(trange[0], '%Y-%m-%d %H:%M:%S')
     end_time = datetime.strptime(trange[1], '%Y-%m-%d %H:%M:%S')

     # Expand the time range: Add 2 minute each time
     extended_start = start_time - timedelta(minutes=2)
     extended_end = end_time + timedelta(minutes=2)
        # Adds the extended time range to the time ranges
     time_range=[extended_start.strftime('%Y-%m-%d %H:%M:%S'), extended_end.strftime('%Y-%m-%d %H:%M:%S')]
     Mag_data=Read_Mag_npydata(time_range)
     Time_range_s = datetime.strptime(trange[0], "%Y-%m-%d %H:%M:%S")  # 此处可更改
     Time_range_e = datetime.strptime(trange[1], "%Y-%m-%d %H:%M:%S")
     index = np.where(Mag_data.Time < Time_range_s)[0]
     B1=[np.mean(Mag_data.Bx[index]),np.mean(Mag_data.By[index]),np.mean(Mag_data.Bz[index])]
     index = np.where(Mag_data.Time > Time_range_e)[0]
     B2 = [np.mean(Mag_data.Bx[index]), np.mean(Mag_data.By[index]), np.mean(Mag_data.Bz[index])]
     shear_angle = angle_vec(B1, B2)
     Time_range_s = datetime.strptime(time_range[0], "%Y-%m-%d %H:%M:%S")  # 此处可更改
     Time_range_e = datetime.strptime(time_range[1], "%Y-%m-%d %H:%M:%S")
     index = np.where((Mag_data.Time < Time_range_e ) & (Mag_data.Time > Time_range_s))[0]
     MP_MET = np.mean(Mag_data.MET[index])
     MP_X = np.mean(Mag_data.Xmso[index])
     MP_Y = np.mean(Mag_data.Ymso[index])
     MP_Z = np.mean(Mag_data.Zmso[index])
     MP_pos=[MP_X,MP_Y,MP_Z]
     if type=="mp_in":
         B_msh = B1
         B_msp = B2
     else:
         B_msh = B2
         B_msp = B1
     B_MSM = np.vstack((Mag_data.Bx, Mag_data.By, Mag_data.Bz)).T
     B_mva,v,w = pyspedas.minvar(B_MSM)
     flux_vec,MET_diff,flux_t=get_fluxmap_vector(trange[1],MP_MET)
     if not isinstance(flux_vec, int):
         angle_Lf = angle_vec(v[:, 0], flux_vec)
     else:
         angle_Lf=-1
     return B_msh,B_msp,angle_Lf,shear_angle,MET_diff,MP_pos,v[:, 0],flux_t
def get_fluxmap_vector(Time,MET):
    DOY = day_of_year(int(Time[0:4]), int(Time[5:7]), int(Time[8:10]))
    tabfilepath = MESSENGER_data_folder + '/MESSENGER_Data/FIPS/DDR/Fluxmap/'  #
    filename = 'FIPS_FLUXMAP_'+Time[0:4]+str(DOY).zfill(3)+'_DDR_*.xml'
    if not glob.glob(tabfilepath + filename):
        return -1,-1
    print(glob.glob(tabfilepath + filename)[0])
    data = pds4_tools.read(glob.glob(tabfilepath + filename)[0])
    flux_array = data[1]['FLUX']
    flux_MET = (data[1]['START_MET']+data[1]['START_MET'])/2.0
    rows, cols = flux_array.shape
    flux_array = flux_array.reshape(rows, 18, 36)
    index=np.where(np.abs(flux_MET-MET) == np.min(np.abs(flux_MET-MET)))[0][0]
    MET_t=flux_MET[index]
    flux_t = flux_array[index,:,:]
    MET_diff=abs(MET_t-MET)
    index = np.where(flux_t > -1)
    lat = (18-np.mean(index[0]))*10-95
    lon = np.mean(index[1])*10-175
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    # 计算XYZ坐标
    X = np.cos(lat_rad) * np.sin(lon_rad)
    Y = np.cos(lat_rad) * np.cos(lon_rad)
    Z = np.sin(lat_rad)
    flux_vec=[X,Y,Z]
    return flux_vec,MET_diff,flux_t

def SW_aberration(UTC,dis=False):
    # 将本地时间转换为世界时（UTC）
    sk = [
        'G:\Mercury\MESSENGER\IDL\paper1\Bow_shock\IDL_Project\Other_pro\icy\data\spice/naif\generic_kernels\lsk/naif0012.tls',
        'G:\Mercury\MESSENGER\IDL\paper1\Bow_shock\IDL_Project\Other_pro\icy\data\spice/naif\generic_kernels\pck\pck00010.tpc',
        'G:\Mercury\MESSENGER\IDL\paper1\Bow_shock\IDL_Project\Other_pro\icy\data\spice/naif\generic_kernels\spk\planets\de430.bsp']
    spiceypy.furnsh(sk)
    et = spiceypy.utc2et(UTC)
    targ = 'Mercury'
    ref = 'ECLIPJ2000'
    obs = 'Sun'
    abcorr = 'NONE'
    pos2vel = spiceypy.spkezr(targ, et, ref, abcorr, obs)
    position = pos2vel[0][0:3]
    velocity = pos2vel[0][3:6]
    position = position / 149597870
    dis_sun = np.linalg.norm(position)
    v_radial = sum(velocity * position / dis_sun)
    spiceypy.kclear()
    if dis == True:
        return dis_sun
def angle_vec(a, b):
    """
    计算三维矢量之间的夹角。

    参数:
    a -- 第一个三维矢量 (a_x, a_y, a_z)
    b -- 第二个三维矢量 (b_x, b_y, b_z)

    返回值:
    夹角（度）
    """
    # 计算点积
    dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    # 计算范数
    norm_a = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    norm_b = np.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    # 计算夹角余弦值
    cos_theta = dot_product / (norm_a * norm_b)
    # 限制cos_theta在[-1, 1]之间
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # 计算夹角（弧度）
    angle_rad = np.arccos(cos_theta)
    # 转换为角度
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def plot_mp_thickness():
    # 加载数据
    data = np.load(MESSENGER_data_folder + '/mplist.npy', allow_pickle=True)
    # 定义时间格式
    time_format = "%Y-%m-%d %H:%M:%S"

    # 初始化一个列表来存储时间差
    time_diffs = []
    indices = np.where(data['num'] == 'm')[0]
    print(len(indices))
    # 遍历数据计算时间差
    for item in data:
        start_time = datetime.strptime(item['start'], time_format)
        end_time = datetime.strptime(item['end'], time_format)
        time_diff = (end_time - start_time).total_seconds()
        time_diffs.append(time_diff)
    # 提取需要的字段
    thickness_mp = data['thickness_mp'][indices]*2440#time_diffs#
    length= np.sqrt(data['y_msm']**2 + data['z_msm']**2)
    dis_sun = data['x_msm'][indices]
    x_MSM = length[indices]#data['z_msm']

    # 设置绘图参数
    dis_sun_bins = np.linspace(-4.5, 3.5, 40 + 1)  # 20 个 bin
    x_MSM_bins = np.linspace(-4.5, 3.5, 40 + 1)  # 80 个 bin

    # 对数据进行二维直方图统计，并计算每个 bin 中 thickness_mp 的平均值
    H, xedges, yedges = np.histogram2d(dis_sun, x_MSM, bins=[dis_sun_bins, x_MSM_bins], weights=thickness_mp)

    # 计算每个 bin 中的平均值
    counts, _, _ = np.histogram2d(dis_sun, x_MSM, bins=[dis_sun_bins, x_MSM_bins])
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = np.nan_to_num(H / counts)



    # 定义彩色颜色映射范围和 Normalize 对象
    norm = mcolors.Normalize(vmin=0, vmax=500)
    cmap = plt.cm.jet  # 使用彩色映射，例如 jet

    # 将没有数据的 bin 的值设置为 NaN
    Z[counts < 3 ] = np.nan

    # 设置彩色映射，白色用于表示没有数据的 bin
    cmap.set_bad(color='white')

    # 绘制伪彩图，并设置彩色 colorbar
    plt.figure(figsize=(10, 6))
    plt.imshow(Z.T, extent=[dis_sun_bins[0], dis_sun_bins[-1], x_MSM_bins[0], x_MSM_bins[-1]],
               aspect='auto', origin='lower', cmap=cmap, norm=norm)
    plt.colorbar(label='Thickness/km', extend='both')

    # 设置标签和标题
    plt.xlabel('x_MSM')
    plt.ylabel('yz_MSM')
    plt.title('Thickness')



    # 显示图形
    plt.show()
def Read_FIPS_TABdata(filename,mode='CDR',type='scan'):

    if mode == 'CDR' and type == 'scan':
        structure = pds4_tools.pds4_read(filename)
        head=structure[0]
        scan_data=structure[1]
        #get E/Q Table
        # 检查数据属性
        if hasattr(head, 'data'):
            data = head.data
            # 将字节数据解码为字符串
            decoded_data = data.decode('utf-8', errors='ignore')
            # 查找包含 "E/Q Table used" 的行
            lines = decoded_data.split('\n')
            for line in lines:
                if "E/Q Table used" in line:
                    EQ_Table_filename=line.strip()[18:34]
                    break
        #获得Energy Table文件
        calibfilepath = MESSENGER_data_folder + '/MESSENGER_Data\FIPS\CDR/calibration/'
        calibfull_path = glob.glob(calibfilepath+EQ_Table_filename+'_*.xml')[0]
        if os.path.isdir(calibfilepath):
            if os.path.exists(calibfull_path):
                EQ_Table=pds4_tools.pds4_read(calibfull_path)[1]
            else:
                Download_MESSENGER_FIPS_Data(type='calibration')
                EQ_Table = pds4_tools.pds4_read(calibfull_path)[1]
        else:
            os.makedirs(calibfilepath)
            Download_MESSENGER_FIPS_Data(type='calibration')
            EQ_Table = pds4_tools.pds4_read(calibfull_path)[1]
        EQ_Table_head=pds4_tools.pds4_read(calibfull_path)[0]
        outdtype = [ ('Time', 'datetime64[ms]'),
                     ('MET', 'float32'),
                     ('Energy','float32',(64,)),
                     ('PFlux','float32',(64,)),
                     ('Counts', 'float32', (64,)),
                     ('PSD', 'float32', (64,)),
                     ('Scan_Time','float32'),
                     ('Accum_Time','float32'),
                    # ('N_proton','float32'),
                    # ('T_proton','float32'),
                    # ('K_proton','float32'),
                     ('n_obs','float32'),
                     ('v_obs','float32')]
        n = scan_data['TIME'].size
        out = np.recarray(n, outdtype)
        date_str = scan_data['TIME'][0]
        year = int(date_str[:4])
        doy = int(date_str[5:8])
        Counts_filename = (MESSENGER_data_folder + '/MESSENGER_Data\FIPS\EDR/TAB/' +
                           'FIPS_R' + str(year) + str(doy).zfill(3) + 'EDR_V1.xml')
        Counts_data = Read_FIPS_TABdata(filename=Counts_filename, mode='EDR', type='Counts')
       # Fluxmap_filename = (MESSENGER_data_folder + '/MESSENGER_Data\FIPS\DDR/TAB/' +
                          # 'FIPS_FLUXMAP_' + str(year) + str(doy).zfill(3) + '_DDR_V*.xml')
        #Fluxmap_data = Read_FIPS_TABdata(filename=Fluxmap_filename[0], mode='DDR', type='Fluxmap')
        for i in range(0, n):
            date_str = scan_data['TIME'][i]
            year = int(date_str[:4])
            doy = int(date_str[5:8])
            hms = date_str[9:23]
            ymd = doy_to_date(year,doy)
            time_str = ymd+' '+hms
            out.Time[i] = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            out.Energy[i] = EQ_Table['EQ_TABLE_'+str(scan_data['FIPS_SCANTYPE'][i])]
            out.Accum_Time[i]=EQ_Table['ACCUM_TIME_'+str(scan_data['FIPS_SCANTYPE'][i])][1]
            Counts_MET = np.array(Counts_data.field('MET'))
            indices = np.where(Counts_MET == scan_data['MET'][i])
            if indices[0].size != 0:
              out.Counts[i] = Counts_data.field('PROTON_RATE')[indices][0]
            else:
              out.Counts[i] = np.zeros(64)
            out.PFlux[i] = scan_data['PROTON_DIFFINTENS'][i]
            m_proton=1.6726e-27
            v_proton=np.sqrt(2*out.Energy[i]*1.60218e-16/m_proton)
            out.PSD[i]=out.PFlux[i]*6.24e19*(m_proton/v_proton**2)
            #indices = np.where( out.PSD[i] > 0 )
            # 对速度和 PSD 数据进行排序
            sorted_indices = np.argsort(v_proton)  # 获取升序排列的索引
            v_proton_sorted = v_proton[sorted_indices]
            PSD_sorted = out.PSD[i][sorted_indices]

            out.n_obs[i]= np.trapz(PSD_sorted, v_proton_sorted)

            if out.n_obs[i] > 0:
               out.v_obs[i] = np.trapz(v_proton_sorted*PSD_sorted, v_proton_sorted)/out.n_obs[i]
            else:
                out.v_obs[i]=0
                #out.PSD_mean[i] = np.mean(out.PSD[i,indices])
            #out.Energy_mean[i] = np.sum(out.PSD[i,indices]*out.Energy[i,indices])/np.sum(out.PSD[i,indices])
            #NTK=FitKappaDist.FitKappaDist(v=v_proton,f=out.PSD[i],n0=10000000,T0=10e7,Counts=out.Counts[i])
            #out.N_proton[i] = NTK[0] * 10e-6
            #out.T_proton[i] = NTK[1]
            #out.K_proton[i] = NTK[2]
            #scan time
            if hasattr(EQ_Table_head, 'data'):
                data = EQ_Table_head.data
                # 将字节数据解码为字符串
                decoded_data = data.decode('utf-8', errors='ignore')
                # 查找包含 "scan time" 的行
                lines = decoded_data.split('\n')
                for line in lines:
                    if 'Table(s) '+str(scan_data['FIPS_SCANTYPE'][i])+' have a total scan time' in line:
                        SCAN_resolution = float(line.strip()[39:44])
                        out.Scan_Time[i]=SCAN_resolution
                        break

        out.MET = scan_data['MET']
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data\FIPS\CDR/NPY/'
        if os.path.isdir(npyfilepath):
            np.save(npyfilepath + 'FIPS_R' + str(year) + str(doy).zfill(3) + 'CDR.npy', out)
        else:
            os.makedirs(npyfilepath)
            np.save(npyfilepath + 'FIPS_R' + str(year) + str(doy).zfill(3) + 'CDR.npy', out)

    if mode == 'EDR' and type == 'Counts':
        structure = pds4_tools.pds4_read(filename)
        EDR_data = structure[0]
        return EDR_data
    if mode == 'DDR' and type == 'Fluxmap':
        structure = pds4_tools.pds4_read(filename)
        fluxmap_data = structure[1]
        outdtype = [('START_MET', 'float32'),
                    ('END_MET', 'float32'),
                    ('Fluxmap', 'float32', (648,))]
        n = fluxmap_data['MET'].size
        out = np.recarray(n, outdtype)
        out.START_MET=fluxmap_data['START_MET']
        out.END_MET = fluxmap_data['END_MET']
        out.Fluxmap = fluxmap_data['FLUX']
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data\FIPS\DDR/Fluxmap/NPY/'
        if os.path.isdir(npyfilepath):
            np.save(npyfilepath + 'FIPS_FLUXMAP_' + str(year) + str(doy).zfill(3) + '_DDR.npy', out)
        else:
            os.makedirs(npyfilepath)
            np.save(npyfilepath + 'FIPS_FLUXMAP_' + str(year) + str(doy).zfill(3) + '_DDR.npy', out)
def Read_FIPS_npydata(Time_Range,mode='CDR',type='scan'):
    if mode == 'CDR' and type == 'scan':
        from MESSENGER_GUI import day_of_year
        DOY_start = day_of_year(int(Time_Range[0][0:4]), int(Time_Range[0][5:7]), int(Time_Range[0][8:10]))
        DOY_end = day_of_year(int(Time_Range[1][0:4]), int(Time_Range[1][5:7]), int(Time_Range[1][8:10]))
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data/FIPS/CDR/NPY/'
        if not os.path.isdir(npyfilepath):
            os.makedirs(npyfilepath)
            # 若时间范围跨多个文件则将连接多个文件
        FIPS_SCAN = {}  # 创建连接多个npy文件数据的空字典
        for i in range(DOY_start, DOY_end + 1):
            npyfilename = 'FIPS_R' + Time_Range[0][0:4] + str(i).zfill(3) + 'CDR.npy'
            npyfull_path = os.path.join(npyfilepath, npyfilename)
            # 检查文件是否存在
            if os.path.exists(npyfull_path):
                cdrscan_npy = np.load(npyfull_path, allow_pickle=True)
                print("cdr scan.npy loaded successfully.")
            else:
                print("cdr scan.npy does not exist,and will attempt to convert TAB. data to npy. data.")
                return -1
                tabfilepath = MESSENGER_data_folder +  '/MESSENGER_Data/FIPS/CDR/TAB/'
                if not os.path.isdir(tabfilepath):
                    os.makedirs(tabfilepath)
                tabfilename = 'FIPS_R' + Time_Range[0][0:4] + str(i).zfill(3) + 'CDR_*.xml'
                if glob.glob(tabfilepath + tabfilename):
                    tabfull_path = glob.glob(tabfilepath + tabfilename)[0]
                    print("cdr scan.tab loaded successfully and will be transfered to npy.file.")
                    Read_FIPS_TABdata(tabfull_path)
                    print(tabfull_path)
                    cdrscan_npy = np.load(npyfull_path, allow_pickle=True)
                else:
                    print("cdr scan.tab does not exist,and will be download from PDS.")
                    Download_MESSENGER_FIPS_Data(int(Time_Range[0][0:4]),i)
                    Download_MESSENGER_FIPS_Data(int(Time_Range[0][0:4]),i,mode='EDR',type='Counts')
                    tabfull_path = glob.glob(tabfilepath + tabfilename)[0]
                    Read_FIPS_TABdata(tabfull_path, mode)
                    cdrscan_npy = np.load(npyfull_path, allow_pickle=True)
            for name in cdrscan_npy.dtype.names:
                # 如果数组名已经存在于 merged_arrays 中，则将当前数组与已存在的数组合并
                if name in FIPS_SCAN:
                    FIPS_SCAN[name] = np.concatenate((FIPS_SCAN[name], cdrscan_npy[name]))
                # 如果数组名不存在于 merged_arrays 中，则将当前数组添加到 merged_arrays 中
                else:
                    FIPS_SCAN[name] = cdrscan_npy[name]
        Time_range_s = datetime.strptime(Time_Range[0], "%Y-%m-%d %H:%M:%S")  # 此处可更改
        Time_range_e = datetime.strptime(Time_Range[1], "%Y-%m-%d %H:%M:%S")
        index = np.where((FIPS_SCAN['Time'] > Time_range_s) & (FIPS_SCAN['Time'] < Time_range_e))
        # 如果索引为空，则返回字符串 'No FIPS SCAN data'
        if len(index[0]) == 0:
            return 'No FIPS SCAN data'
        returndtype = [('Time', 'datetime64[ms]'),
                    ('MET', 'float32'),
                    ('Energy', 'float32', (64,)),
                    ('PFlux', 'float32', (64,)),
                    ('Counts', 'float32', (64,)),
                    ('PSD', 'float32', (64,)),
                    ('Scan_Time', 'float32'),
                    ('Accum_Time', 'float32'),
                    #('N_proton', 'float32'),
                    #('T_proton', 'float32'),
                    #('K_proton', 'float32'),
                    ('n_obs', 'float32'),
                    ('v_obs', 'float32')]
        returndata = np.recarray(len(index[0]), returndtype)
        returndata.Time = FIPS_SCAN['Time'][index[0]]
        returndata.Energy = FIPS_SCAN['Energy'][index[0]]
        returndata.PFlux = FIPS_SCAN['PFlux'][index[0]]
        returndata.PSD = FIPS_SCAN['PSD'][index[0]]
        returndata.Scan_Time = FIPS_SCAN['Scan_Time'][index[0]]
        returndata.MET = FIPS_SCAN['MET'][index[0]]
        returndata.Counts = FIPS_SCAN['Counts'][index[0]]
        returndata.Accum_Time = FIPS_SCAN['Accum_Time'][index[0]]
       # returndata.N_proton = FIPS_SCAN['N_proton'][index[0]]
        #returndata.T_proton = FIPS_SCAN['T_proton'][index[0]]
        #returndata.K_proton = FIPS_SCAN['K_proton'][index[0]]
        returndata.n_obs = FIPS_SCAN['n_obs'][index[0]]
        returndata.v_obs = FIPS_SCAN['v_obs'][index[0]]
        return returndata
def doy_to_date(year, day_of_year):
    # How many days of the year are converted to dates
    # The first day of the specified year
    start_of_year = datetime(year, 1, 1)
    # Calculate the target date by increasing the number of days
    target_date = start_of_year + timedelta(days=int(day_of_year) - 1)
    # Format the date as YYYY-MM-DD
    return target_date.strftime('%Y-%m-%d')
def Download_MESSENGER_FIPS_Data(year=None, doy=None, mode='CDR',type='SCAN',All_data='False'):
    if mode=='CDR' and type=='calibration':
        URL = 'https://search-pdsppi.igpp.ucla.edu/ditdos/download?id=pds://PPI/mess-epps-fips-calibrated/calibration'
        download_dir = MESSENGER_data_folder + '/MESSENGER_Data\FIPS\CDR/calibration/'
        # 发送HTTP GET请求下载ZIP文件
        response = requests.get(URL)
        response.raise_for_status()
        # 读取ZIP文件内容
        zip_data = io.BytesIO(response.content)
           # 解压缩ZIP文件到指定文件夹
        zip_ref=zipfile.ZipFile(zip_data, 'r')
        zip_ref.extractall(download_dir)
        # 添加一个语句来结束当前函数
        return "Download and extraction completed successfully"
    # Download MESSENGER_MAG_DATA,
    if All_data == 'True':
        years = [2011, 2012, 2013, 2014, 2015]
        doys = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306,
                336, 1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 1, 32, 60, 91, 121, 152, 182, 213, 244, 274,
                305, 335, 1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        for i in range(0, 5):
            for j in range(12):
                try:
                   Download_MESSENGER_FIPS_Data(year=years[i], doy=doys[i * 12 + j], mode='DDR', type='Fluxmap')
                   # Download_MESSENGER_FIPS_Data(year=years[i], doy=doys[i * 12 + j])
                   print(i, j + 1)
                except zipfile.BadZipFile:
                    print("文件不是一个有效的 ZIP 文件，跳过此文件。")
                    continue  # 继续下一个循环
    Month = int(doy_to_date(year, doy)[5:7])
    # Get the first day of the month
    first_day = datetime(year, Month, 1)
    # Get the first day of the next month
    if Month == 12:
        next_month_first_day = datetime(year + 1, 1, 1)
    else:
        next_month_first_day = datetime(year, Month + 1, 1)
    # Get the last day of the month
    last_day = next_month_first_day - timedelta(days=1)
    # Calculate the first day of the month and the last day of the year
    first_day_of_year = datetime(year, 1, 1)
    first_day_of_year_number = (first_day - first_day_of_year).days + 1
    last_day_of_year_number = (last_day - first_day_of_year).days + 1
    first_last_month_str = str(first_day_of_year_number).zfill(3) + '_' + str(last_day_of_year_number).zfill(3) + '_'
    Month_str = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    # # Get download link
    if mode=='CDR' and type=='SCAN':
        URL = 'https://search-pdsppi.igpp.ucla.edu/ditdos/download?f=zip&id=pds://PPI/mess-epps-fips-calibrated/data/scan/' + str(
            year) + '/' + first_last_month_str + Month_str[Month - 1]
        zipfilename = '/'+ 'FIPS_R' + str(year)[0:4] + str(doy).zfill(3) + 'CDR_V5'
        download_dir = MESSENGER_data_folder + '/MESSENGER_Data\FIPS\CDR/TAB/'
        # 发送HTTP GET请求下载ZIP文件
        response = requests.get(URL)
        response.raise_for_status()
        # 读取ZIP文件内容
        zip_data = io.BytesIO(response.content)
           # 解压缩ZIP文件到指定文件夹
        zip_ref=zipfile.ZipFile(zip_data, 'r')
        zip_ref.extractall(download_dir)
    if mode=='EDR' and type=='Counts':
        URL = 'https://search-pdsppi.igpp.ucla.edu/ditdos/download?id=pds://PPI/mess-epps-fips-raw/data/scan/' + str(
            year) + '/' + first_last_month_str + Month_str[Month - 1]
        zipfilename = '/'+ 'FIPS_R' + str(year)[0:4] + str(doy).zfill(3) + 'CDR_V5'
        download_dir = MESSENGER_data_folder + '/MESSENGER_Data\FIPS\EDR/TAB/'
        # 发送HTTP GET请求下载ZIP文件
        response = requests.get(URL)
        response.raise_for_status()
        # 读取ZIP文件内容
        zip_data = io.BytesIO(response.content)
           # 解压缩ZIP文件到指定文件夹
        zip_ref=zipfile.ZipFile(zip_data, 'r')
        zip_ref.extractall(download_dir)
    if mode=='DDR' and type=='Fluxmap':
        URL = 'https://search-pdsppi.igpp.ucla.edu/ditdos/download?id=pds://PPI/mess-epps-fips-derived/data/fluxmap/'+ str(
            year) + '/' + first_last_month_str + Month_str[Month - 1]
        download_dir = MESSENGER_data_folder + '/MESSENGER_Data\FIPS\DDR/Fluxmap/'
        # 发送HTTP GET请求下载ZIP文件
        response = requests.get(URL)
        response.raise_for_status()
        # 读取ZIP文件内容
        zip_data = io.BytesIO(response.content)
        # 解压缩ZIP文件到指定文件夹
        zip_ref = zipfile.ZipFile(zip_data, 'r')
        zip_ref.extractall(download_dir)

def transfer_tab():
 directory_path = r'G:\SpaceScience\MESSENGER\MESSENGER_Data\FIPS\CDR\TAB'

# 使用 glob 模块递归搜索目录下的所有 .xml 文件
 xml_files = glob.glob(os.path.join(directory_path, '**', '*.xml'), recursive=True)

# 提取文件名而不是完整路径
 file_names = [os.path.basename(file) for file in xml_files]

# 输出文件名
 print("XML files found:")

 for i in range(1321,len(file_names)):
    print(i)#1320
    file='G:\SpaceScience\MESSENGER\MESSENGER_Data\FIPS\CDR\TAB/'+file_names[i]
    Read_FIPS_TABdata(filename=file)
def plot_shear_fL():
    mp_path = MESSENGER_data_folder + '/mplist.npy'
    mp_data = np.load(mp_path, allow_pickle=True)
    print(len(mp_data['angle_Lf']))
    for i in range(0,len(mp_data['angle_Lf'])) :
        print(mp_data['angle_Lf'][i])
        if mp_data['angle_Lf'][i] > 90:
            mp_data['angle_Lf'][i]=180-mp_data['angle_Lf'][i]
        # 网格化的参数
    angle_Lf_bins = np.arange(0, 91, 10)  # 从0到90，每10度一个网格
    shear_angle_bins = np.arange(0, 181, 10)  # 从0到180，每10度一个网格

    # 计算每个网格的元素数量
    hist, xedges, yedges = np.histogram2d(
            mp_data['shear_angle'],
            mp_data['angle_Lf'],
            bins=[shear_angle_bins, angle_Lf_bins]
        )
    # 绘制伪彩图
    plt.figure(figsize=(10, 6))

    # 创建伪彩图
    plt.imshow(hist.T, extent=[shear_angle_bins[0], shear_angle_bins[-1], angle_Lf_bins[0], angle_Lf_bins[-1]],
               origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')

    # 添加颜色条
    plt.colorbar(label='number')

    # 添加标签
    plt.xlabel('Shear Angle (°)')
    plt.ylabel('Angle Lf (°)')

    # 显示图像
    plt.show()
        #Download_MESSENGER_FIPS_Data( mode='DDR', type='Fluxmap', All_data='True')
def plot_on_spherical_mesh(flux_data):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    plasma_cmap = plt.get_cmap('plasma')
    # Create a new colormap that has black for NaN values
    colors = plasma_cmap(np.arange(plasma_cmap.N))
    colors[:1, :] = 0.5  # Make the first color black
    new_cmap = ListedColormap(colors)
    # Avoid log(0) by setting a minimum flux value (epsilon)
    epsilon = np.min(flux_data[(flux_data>0)])-np.min(flux_data[(flux_data>0)])/2
    flux_data[(flux_data ==0)] = epsilon
    # Define the spherical coordinates
    theta = np.linspace(-np.pi/2, np.pi/2, 18)  # 0 to 180 degrees in radians (latitude)
    phi = np.linspace(-np.pi,  np.pi, 36)  # 0 to 360 degrees in radians (longitude)
    phi, theta = np.meshgrid(phi, theta)  # Note the order here
    b_test = np.array([-28,0,20])
    theta_test = np.arccos(b_test[2]/np.linalg.norm(b_test))
    phi_test = np.arctan2(b_test[1],b_test[0])
    # Plot the data using pcolormesh
    fig, ax = plt.subplots(subplot_kw={'projection': 'mollweide'})
    #ax.scatter(np.pi-phi_test,np.pi/2-theta_test,color='red',s=20)
    pcm = ax.pcolormesh(phi, theta, flux_data, norm=LogNorm(vmin = 1E9,vmax=1E12), cmap=new_cmap, shading='auto')
    # Add a color bar which maps values to colors
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('H+ Flux (m$^{-2}$s$^{-1}$)')
    # Customize the plot
    #ax.set_xlabel('Longitude')
    #ax.set_ylabel('Latitude')
    #ax.set_title('Flux Map')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Add custom labels
    # Equator and meridian labels
    ax.text(np.pi / 2,0, '+$Y_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.text(-np.pi / 2,0, '-$Y_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.text(-np.pi, 0, '-$X_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.text(np.pi, 0, '-$X_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.text(0,0,'+$X_{MSO}$',horizontalalignment='center', verticalalignment='center', fontsize=12)
    # Pole labels
    ax.text(0, np.pi / 2, '+$Z_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
    ax.text(0, -np.pi / 2, '-$Z_{MSO}$', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
    plt.show()
    def custom_grid(ax):
        theta = np.linspace(0, np.pi, 18)
        phi = np.linspace(0,2*np.pi, 36)
        offsets_1 = [0,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4]
        offsets_2 = [0,np.pi/4,-np.pi/4]
        for i in offsets_1:
            ax.plot(np.zeros(len(theta))+i,np.pi/2-theta,color='gray',alpha=.8,linewidth=.7)
        for i in offsets_2:
            ax.plot(np.pi-phi,np.zeros(len(phi))+i,color='gray',alpha=.8,linewidth=.7)
    custom_grid(ax)
    #ax.grid(True)
    return ax


##transfer_tab()
#plot_mp_thickness()
#Read_Sun_MP_BS()
#Plot_GUI(time_ranges=[('2012-07-30 23:40:00','2012-07-30 23:59:00')],time_list=None)
Plot_GUI(time_list='MP_Sun')
# 生成包含 -1 和 0 的示例数据
#trange=['2012-07-30 23:50:00','2012-07-30 23:55:00']
#B_msh,B_msp,angle_Lf,shear_angle,MET_diff,MP_pos,v,flux_t=get_mp_parameter(trange,method='mva',type=None)
#plot_on_spherical_mesh(flux_t)
# 定义 XML 文件所在的目录路径









