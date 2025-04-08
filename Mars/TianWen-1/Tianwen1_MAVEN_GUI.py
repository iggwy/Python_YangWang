
import os
import time
import requests
from datetime import datetime
from tqdm import tqdm

current_path = os.path.abspath(__file__)
# Get the path to the folder where the current script file resides
MARs_data_folder = os.path.dirname(current_path)



def download_TW1_data(instru='MAMOG', accur='32Hz', Time_range=None):
    """
    下载指定 Hz MOMAG 数据文件。
    如果本地 TXT 文件 TW1_MOMAG_filename&id.txt 不存在，则从 GitHub 下载。
    然后解析文件名并下载数据。
    """
    TXT_FILE = "TW1_MOMAG_filename&id.txt"
    SAVE_DIR = f"TIANWEN1_Data/MOMAG/{accur}/"
    GITHUB_URL = "https://raw.githubusercontent.com/iggwy/Python_YangWang/main/Mars/TianWen-1/TW1_MOMAG_filename&id.txt"
    # 2C 下载链接模板
    BASE_URL_2C = (
        "https://moon.bao.ac.cn/web/zhmanager/kxsj"
        "?p_p_id=scientificdata_WAR_ScientificDataportlet"
        "&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view"
        "&p_p_cacheability=cacheLevelPage&p_p_col_id=column-1"
        "&p_p_col_count=1&p_p_resource_id=download&id={id_2c}&flagCol=SP&name={name_2c}"
    )

    # 2CL 下载链接模板
    BASE_URL_2CL = (
        "https://moon.bao.ac.cn/web/zhmanager/kxsj"
        "?p_p_id=scientificdata_WAR_ScientificDataportlet"
        "&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view"
        "&p_p_cacheability=cacheLevelPage&p_p_col_id=column-1&p_p_col_count=1"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3Bp_p_col_id=column-1"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3Bp_p_lifecycle=2"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3Bname={name_2c}"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3Bp_p_state=normal"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3Bp_p_col_count=1"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3Bp_p_resource_id=download"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3Bid={id_2c}"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3BflagCol=SPL"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3Bp_p_mode=view"
        "&_scientificdata_WAR_ScientificDataportlet_amp%3Bp_p_cacheability=cacheLevelPage"
        "&p_p_resource_id=download&id={id_2cl}&flagCol=SPL&name={name_2cl}"
    )

    # HTTP headers with cookie for authentication
    HEADERS = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB-oxendict;q=0.7,en-GB;q=0.6,en-US;q=0.5",
        "connection": "keep-alive",
        "content-length": "30",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "host": "moon.bao.ac.cn",
        "origin": "https://moon.bao.ac.cn",
        "referer": "https://moon.bao.ac.cn/web/zhmanager/kxsj?missionName=HX1&zhName=MOMAG&grade=2C",
        "sec-ch-ua": '"Microsoft Edge";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
        "x-requested-with": "XMLHttpRequest",
    }

    def login_and_get_cookies(username, password):
        import os
        import time
        import requests
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        COOKIE_FILE = "cookies.txt"#cookies文件，如果不存在或cookies内容失效将会通过登录自动下载
        TEST_URL = "https://moon.bao.ac.cn/web/zhmanager/kxsj?missionName=HX1&zhName=MOMAG&grade=2C"
        # HTTP headers with cookie for authentication
        HEADERS = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB-oxendict;q=0.7,en-GB;q=0.6,en-US;q=0.5",
            "connection": "keep-alive",
            "content-length": "30",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "host": "moon.bao.ac.cn",
            "origin": "https://moon.bao.ac.cn",
            "referer": "https://moon.bao.ac.cn/web/zhmanager/kxsj?missionName=HX1&zhName=MOMAG&grade=2C",
            "sec-ch-ua": '"Microsoft Edge";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
            "x-requested-with": "XMLHttpRequest",
        }

        # === 保存 cookies 为 TXT 文件 ===
        def save_cookies(cookie_dict):
            with open(COOKIE_FILE, "w", encoding="utf-8") as f:
                for key, value in cookie_dict.items():
                    f.write(f"{key}={value}\n")

        # === 读取 cookies 从 TXT 文件 ===
        def load_cookies():
            if not os.path.exists(COOKIE_FILE):
                return None

            cookie_dict = {}
            with open(COOKIE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        cookie_dict[key] = value

            return cookie_dict

        # === 检查是否已登录 ===
        def is_logged_in(cookies):
            try:
                response = requests.get(TEST_URL, cookies=cookies, headers=HEADERS)
                return 'id="logoutBtn"' in response.text
            except Exception as e:
                print("测试登录状态失败：", e)
                return False

        # === 登录并获取 cookies ===
        def login_and_get_cookies(username, password):
            driver = webdriver.Edge()
            driver.get(TEST_URL)

            wait = WebDriverWait(driver, 3)

            login_trigger = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "登录")))
            login_trigger.click()

            username_input = wait.until(EC.element_to_be_clickable((By.ID, "_58_login")))
            username_input.clear()
            username_input.send_keys(username)

            password_input = wait.until(EC.element_to_be_clickable((By.ID, "_58_password")))
            password_input.clear()
            password_input.send_keys(password)

            captcha_text = wait.until(EC.visibility_of_element_located((By.ID, "checkCode"))).text
            print(f"验证码识别为：{captcha_text}")
            captcha_input = wait.until(EC.element_to_be_clickable((By.ID, "inputCode")))
            captcha_input.clear()
            captcha_input.send_keys(captcha_text)

            submit_button = wait.until(EC.element_to_be_clickable((By.ID, "login")))
            submit_button.click()

            time.sleep(3)

            cookies = driver.get_cookies()
            cookie_dict = {cookie["name"]: cookie["value"] for cookie in cookies}

            cookie_dict["ce5web_authcode"] = "5308"

            driver.quit()

            print("登录成功，Cookies 已获取：")
            for k, v in cookie_dict.items():
                print(f"{k}: {v}")

            save_cookies(cookie_dict)
            return cookie_dict

        # === 统一接口：如果无效就重新登录 ===
        def get_cookies(username, password):
            cookie_dict = load_cookies()
            if cookie_dict and is_logged_in(cookie_dict):
                print("已有有效的登录 cookie，无需重新登录")
                return cookie_dict

            print("cookie 无效，正在重新登录...")
            return login_and_get_cookies(username, password)

        # === 使用 cookie 访问 URL 示例 ===
        def access_url_with_cookies(url, cookie_file):
            cookies = {}
            with open(cookie_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        cookies[key] = value

            response = requests.get(url, cookies=cookies)

            if response.status_code == 200:
                print("成功访问页面")
                print(response.text[:200])  # 打印部分返回内容
            else:
                print(f"访问失败，状态码: {response.status_code}")

        # ==== 主程序调用 ====
        if __name__ == "__main__":
            username = "yangwangdsel2025"
            password = "Wang1234."

            cookies = get_cookies(username, password)
            return cookies
            print("\n最终获取的 cookies：")
            for k, v in cookies.items():
                print(f"{k}: {v}")

            # 尝试访问页面
            access_url_with_cookies(TEST_URL, COOKIE_FILE)



    def ensure_txt_exists():
        """检查 TXT 文件是否存在，不存在则从 GitHub 下载"""
        if not os.path.exists(TXT_FILE):
            print(f"{TXT_FILE} not found. Downloading from GitHub...")
            try:
                response = requests.get(GITHUB_URL, stream=True)
                response.raise_for_status()
                with open(TXT_FILE, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {TXT_FILE} successfully.")
            except requests.RequestException as e:
                print(f"Failed to download TXT file: {e}")
                return False
        return True
    if not ensure_txt_exists():
        return

    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 读取 TXT 文件
    filenames_and_ids = []
    with open(TXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                filenames_and_ids.append((parts[0], parts[1]))

    # 过滤指定 Hz 数据
    filtered_files = [(filename, file_id) for filename, file_id in filenames_and_ids if accur in filename]

    # 按时间范围筛选数据
    if Time_range:
        start_time = datetime.strptime(Time_range[0], "%Y-%m-%dT%H:%M:%S.%f")
        end_time = datetime.strptime(Time_range[1], "%Y-%m-%dT%H:%M:%S.%f")

        def extract_time_from_filename(filename):
            """从文件名提取开始和结束时间"""
            parts = filename.split('_')
            start_str, end_str = parts[5], parts[6]
            start_datetime = datetime.strptime(start_str, "%Y%m%d%H%M%S")
            end_datetime = datetime.strptime(end_str, "%Y%m%d%H%M%S")
            return start_datetime, end_datetime

        def is_within_time_range(filename):
            """检查文件时间范围是否与指定范围有交集"""
            file_start, file_end = extract_time_from_filename(filename)
            return not (file_end < start_time or file_start > end_time)

        filtered_files = [(filename, file_id) for filename, file_id in filtered_files if is_within_time_range(filename)]

    # 下载文件
    for filename, file_id in tqdm(filtered_files, desc=f"Downloading {accur} Files"):
        download_url_2C = BASE_URL_2C.format(id_2c=file_id, name_2c=filename)
        download_url_2CL = BASE_URL_2CL.format(id_2c=file_id, name_2c=filename,id_2cl=file_id+'0', name_2cl=filename+'L')
        file_path_2C = os.path.join(SAVE_DIR, filename)
        file_path_2CL = os.path.join(SAVE_DIR, filename+'L')
        COOKIES = login_and_get_cookies('yangwangdsel2025', 'Wang1234.')
        # 跳过已存在的文件
        if os.path.exists(file_path_2C):
            print(f"File already exists: {filename}")
            continue
            # 跳过已存在的文件
        if os.path.exists(file_path_2CL):
                print(f"File already exists: {filename+'L'}")
                continue

        # 下载文件
        try:
            response = requests.get(download_url_2C, headers=HEADERS, cookies=COOKIES, stream=True)
            response.raise_for_status()
            with open(file_path_2C, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
            response = requests.get(download_url_2CL, headers=HEADERS, cookies=COOKIES, stream=True)
            response.raise_for_status()
            with open(file_path_2CL, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filename+'L'}")
        except requests.RequestException as e:
            print(f"Failed to download {filename+'L'}: {e}")
# 调用函数
#download_TW1_data(instru='MAMOG', accur='1Hz', Time_range=["2020-08-30T00:00:00.000", "2024-09-01T00:00:00.000"]

def webbugs_filenames():
    """
    爬取网页上的文件名和对应的ID，筛选包含 'HX1-Or_GRAS_MOMAG-DB-1Hz' 或 'HX1-Or_GRAS_MOMAG-DB-32Hz' 的文件，
    并保存到 file_list.txt。
    """
    BASE_URL = "https://moon.bao.ac.cn/web/zhmanager/kxsj"  # 可替换
    PARAMS = {
        "p_p_id": "scientificdata_WAR_ScientificDataportlet",
        "p_p_lifecycle": "2",
        "p_p_state": "normal",
        "p_p_mode": "view",
        "p_p_cacheability": "cacheLevelPage",
        "p_p_col_id": "column-1",
        "p_p_col_count": "1",
        "p_p_resource_id": "searchList",
        "missionName": "HX1",
        "grade": "2C",
        "zhName": "MOMAG",
        "beginDate": "",
        "endDate": "",
        "timeFlag": "0",
        "pageSize": "5",  # 每页记录数，可调整
        "name": "",
        "startQs": "",
        "endQs": "",
        "longitudeBegin": "",
        "longitudeEnd": "",
        "latitudeBegin": "",
        "latitudeEnd": "",
    }

    def extract_filenames_and_ids(json_data):
        """从 JSON 数据中提取符合条件的文件名和对应的 ID"""
        filenames_and_ids = []
        for entry in json_data.get("data", []):
            filename = entry.get("dmName", "")
            file_id = entry.get("dmProductId", "")  # 假设文件ID是dmID字段
            if "HX1-Or_GRAS_MOMAG-DB-1Hz" in filename or "HX1-Or_GRAS_MOMAG-DB-32Hz" in filename:
                filenames_and_ids.append((filename, file_id))
        return filenames_and_ids

    txt_filename = "TW1_MOMAG_filename&id.txt"
    with open(txt_filename, "w", encoding="utf-8") as txtfile:
        total_files = 0
        for page in range(1, 1600):  # 这里只爬取前 5 页，修改为你需要的页数
            print(f"\n正在爬取第 {page} 页...")
            PARAMS["pageNow"] = str(page)

            try:
                response = requests.get(BASE_URL, params=PARAMS, timeout=10)
                response.raise_for_status()
                print(response.text)  # 打印响应内容，帮助调试
                json_data = response.json()
                filenames_and_ids = extract_filenames_and_ids(json_data)

                if filenames_and_ids:
                    print(f"第 {page} 页获取到 {len(filenames_and_ids)} 个文件:")
                    for fname, file_id in filenames_and_ids:
                        print(f"  - filename: {fname}, ID: {file_id}")
                        # 保存到txt文件
                        txtfile.write(f"{fname}\t{file_id}\n")
                    total_files += len(filenames_and_ids)
                else:
                    print(f"第 {page} 页没有符合条件的文件")

            except requests.exceptions.RequestException as e:
                print(f"第 {page} 页请求失败: {e}")

            time.sleep(1)  # 避免请求过快被封禁

    print(f"\n总共爬取到 {total_files} 个文件名和 ID，已保存在 {txt_filename} 中")

# 调用函数
webbugs_filenames()