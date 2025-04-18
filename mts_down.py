import requests
import zipfile

import os
import zipfile
import struct
from datetime import datetime, timedelta
import influxdb_client,  time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from io import BytesIO
import configparser
import json 

"""
下载原始数据
"""
class ParseBinary():
    def __init__(self,folder_path,token,org,bucket,url) -> None:
        self.folder_path = folder_path
        self.token = token  
        self.org = org  
        self.url = url  
        self.bucket = bucket
        self.client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
        with open("mts.json","r",encoding="utf-8") as f:
            self.exists_json = json.load(f)

    def read_tsp_file(self,file_path):
        strs = ""
        with open(file_path, 'r',encoding="utf-8") as file:
            _fl = file.readline().strip()
            print(_fl)
            first_line = _fl.split(",")
            for line in file:
                strs +=line   

        config = configparser.ConfigParser()
        config.read_string("[DEFAULT]\n" + strs)
        self.config = config
        self.headers = first_line
        self.sample_freq = float(first_line[0])

        return config ,first_line

    def read_sts_file(self,file_path):
        data = []
        with open(file_path, 'rb') as file:
            while True:
                buffer = file.read(4)
                if not buffer:
                    break
                value = struct.unpack('f', buffer)[0]
                data.append(value)
        return data

    def convert_start_time(self,start_time):
        dt = datetime.strptime(start_time, "%Y-%m-%d-%H:%M:%S.%f")
        start_time_ns = (dt - datetime(1970, 1, 1)) // timedelta(microseconds=1)
        return start_time_ns * 1000
    
    def write_to_influxDB(self):
        idx = 0
        points = []
        write_api = self.client.write_api(write_options=SYNCHRONOUS)
        for val in self.data:
            point = (
                Point(self.measurement)
                .tag("trigger_at", self.trigger_at)
                .tag("unit", self.headers[-1].replace("\"",""))
                .field(self.node_name, val)
                .time(self.start_time_ns)
            )
            points.append(point)
            idx  += 1

            if idx == self.sample_freq  :

                self.start_time_ns +=  1_000_000_000  # 时间增加 1 秒，单位为纳秒
                idx = 0
                write_api.write(bucket=self.bucket, org="g_lab", record=points)
                write_api.flush()
                points = []
            self.start_time_ns += 100  # 时间增加 100 纳秒，单位为纳秒
        print("*"*120,len(self.data)) 
        if len(points)>0:
            write_api.write(bucket=self.bucket, org="g_lab", record=points)
            write_api.flush()

    def process_files(self,zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            for file in zip_file.namelist():
                if file.endswith('.tsp') and file not in self.exists_json:
                    bagin_time = time.time()
                    self.measurement = file.split("_")[0]
                    self.node_name = file.split("#")[1].split(".")[0]
                    self.trigger_at = file.split("_")[1].split("#")[0]
                    tsp_path = zip_file.extract(file)
                    print(f"file  = {file} | measurement = {self.measurement}| node_name = {self.node_name} | trigger_at = {self.trigger_at} | tsp_path = {tsp_path}")
                    self.read_tsp_file(tsp_path)
                    os.remove(tsp_path)
                    start_time = self.config.get("GpsInfo", "LMT")

                    self.start_time_ns = self.convert_start_time(start_time)
                    print(f"start_time = {start_time} | self.start_time_ns = {self.start_time_ns}")

                    sts_file = file.split(".")[0]+".sts"
                    sts_path = zip_file.extract(sts_file)
                    self.data = self.read_sts_file(sts_path)
                    os.remove(sts_path)
                    self.write_to_influxDB()
                    with open("mts.json","w",encoding="utf-8") as f :
                        self.exists_json[file] = self.headers
                        print(self.exists_json)
                        json.dump(self.exists_json,f)
                    print(f"write cost time   = {time.time() -  bagin_time :.4f} s ")
    
    def parse_binary(self,analyzeId,pyId,ts,http_headers):
        # 发送POST请求获取二进制数据
        payload = {"analyzeId":analyzeId,"phyTs":{pyId:ts}}
        response = requests.post('http://mts2.coinv.com:8091/api/plugin-analyze/callFile', headers=http_headers, json=payload)
        print(payload)
        if response.status_code != 200: return
        
        file_stream = BytesIO(response.content)
        # 将二进制数据保存为zip文件
        with zipfile.ZipFile(file_stream, 'r') as zip_file:
        # 可以进行一系列操作，例如查看 ZIP 文件中的文件列表
            for file in zip_file.namelist():
                if file.endswith('.tsp') and file not in self.exists_json:
                    bagin_time = time.time()
                    self.measurement = file.split("_")[0]
                    self.node_name = file.split("#")[1].split(".")[0]
                    self.trigger_at = file.split("_")[1].split("#")[0]

                    tsp_path = zip_file.extract(file)
                    print(f"file  = {file} | measurement = {self.measurement}| node_name = {self.node_name} | trigger_at = {self.trigger_at} | tsp_path = {tsp_path}")
                    self.read_tsp_file(tsp_path)
                    os.remove(tsp_path)
                    start_time = self.config.get("GpsInfo", "LMT")

                    self.start_time_ns = self.convert_start_time(start_time)
                    print(f"start_time = {start_time} | self.start_time_ns = {self.start_time_ns}")

                    sts_file = file.split(".")[0]+".sts"
                    sts_path = zip_file.extract(sts_file)
                    self.data = self.read_sts_file(sts_path)
                    os.remove(sts_path)
                    self.write_to_influxDB()
                    with open("mts.json","w",encoding="utf-8") as f :
                        self.exists_json[file] = self.headers
                        json.dump(self.exists_json,f)
                    print(f"write cost time   = {time.time() -  bagin_time :.4f} s ")
            

    def run(self):
        for file in os.listdir(self.folder_path):
            if file.endswith('.zip'):
                zip_path = os.path.join(self.folder_path, file)
                self.process_files(zip_path)


class FileMate:
    def __init__(self,analyzeId,authorization) -> None:
        self.analyzeId = analyzeId
        self.token = "ysCDHW20EAL-9PoeX8ZnFN_tHNK8u6XFuOW8RFFDHFUwtxIvu2ZvLNclgHw4QmH6-oMOdfP_FtlBx_f0MV3dCA=="
        self.org = "g_lab"
        self.url = "http://192.168.100.251:38086"
        self.bucket="shauto"
        self.parseBinary  = ParseBinary("/workspace/data/waveFile/",token=self.token,org=self.org,bucket=self.bucket,url=self.url)

        self.base_url = 'http://mts2.coinv.com:8091/'
        # 示例使用方法
        self.headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'Host': 'mts2.coinv.com:8091',
            'Origin': self.base_url,
            'Pragma': 'no-cache',
            'Proxy-Connection': 'keep-alive',
            'Referer': self.base_url,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.36',
            'X-Authorization': authorization 
        }

    def request_point_list(self,bucket,pyId):
        response = requests.get(f'{self.base_url}api/ts/queryPyData?bucket={bucket}&pyKey=inv_file&pyId={pyId}&start=1601481600000&stop=1696953599000&keys=ts,ts,space,fileName,simple_freq,point_number,eu,file_tsval,file_fft,is_alarm&page=0&size=10&sort=_time,desc'
                                ,headers=self.headers)
        if response.status_code == 200:
            data = response.json()

            with open("mts.json","r",encoding="utf-8") as f:
                mts_dict = json.load(f)

            # 第一页数据文件
            if int(data['numberOfElements']) > 0:
                ts = []
                for content in data['content']:
                    prefix_name = content['fields']['fileName'].split(".")[0]
                    if str(prefix_name+".tsp") in mts_dict:
                        print(f" file {prefix_name} continue.     ")
                        continue
                    ts.append(content['ts'])
                print(ts)
                if len(ts)>0:
                    self.parseBinary.parse_binary(analyzeId=self.analyzeId,pyId=content['pyId'],ts=ts,http_headers=self.headers)
            # 剩余数据文件
            for idx in range(int(data['totalPages'])):
                idx_url = f'{self.base_url}api/ts/queryPyData?bucket={bucket}&pyKey=inv_file&pyId={pyId}&start=1601481600000&stop=1696953599000&keys=ts,ts,space,fileName,simple_freq,point_number,eu,file_tsval,file_fft,is_alarm&page={idx + 1}&size=10&sort=_time,desc'
                print(idx_url)
                response = requests.get(idx_url,headers=self.headers)
                if response.status_code == 200:
                    child_data = response.json()
                    if int(child_data['numberOfElements']) > 0:
                        ts = []
                        for content in child_data['content']:
                            prefix_name = content['fields']['fileName'].split(".")[0]
                            if str(prefix_name+".tsp") in mts_dict:
                                print(f" file {prefix_name} continue.     ")
                                continue
                            ts.append(content['ts'])
                        if len(ts)>0:
                            self.parseBinary.parse_binary(analyzeId=self.analyzeId,pyId=content['pyId'],ts=ts,http_headers=self.headers)

    def request_pages(self):
        with open("points.json","r",encoding="utf-8") as f:
            exists_points = json.load(f)
        url = f'{self.base_url}api/analyzes/phyTreeData/{self.analyzeId}'
        print(url)
        response  =requests.get(url,headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            procs = []
            for item in data:
                if item['children'] :
                    for val in item['children']:
                        if val['children'] :
                            for last in val['children']: 
                                begin = time.time()
                                extra = last['extra']
                                if extra['title'] in exists_points : 
                                    print(f"{extra['title']} continue.     ")
                                    continue
                                # if (extra['title'] != "A12-梁体跨中动应变"):continue
                                print(f"bucket or analyzeId = {extra['tenantId']} | pyId = {extra['id']} title = {extra['title']}")
                                self.request_point_list(extra['tenantId'],extra['id'])

                                with open("points.json","w",encoding="utf-8") as f :
                                    exists_points[extra['title']] = extra['key']
                                    json.dump(exists_points,f)
                                print(f"single point cost time {begin - time.time()} s")
                                print("===="*40)
    

class DownMate:
    def __init__(self,analyzeId,authorization) -> None:
        self.analyzeId = analyzeId
        self.token = "ysCDHW20EAL-9PoeX8ZnFN_tHNK8u6XFuOW8RFFDHFUwtxIvu2ZvLNclgHw4QmH6-oMOdfP_FtlBx_f0MV3dCA=="
        self.org = "g_lab"
        self.url = "http://192.168.100.251:38086"
        self.bucket="shuohuang-autooscillation"
        
        self.client = influxdb_client.InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.base_url = 'http://mts2.coinv.com:8091/'
        # 示例使用方法


        self.headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'Host': 'mts2.coinv.com:8091',
            'Origin': self.base_url,
            'Pragma': 'no-cache',
            'Proxy-Connection': 'keep-alive',
            'Referer': self.base_url,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.36',
            'X-Authorization': authorization 
        }
    def down_data(self,pyId,ts,fields):

        url = f"{self.base_url}api/plugin-analyze/call"
        payload = {"analyzeId":self.analyzeId,"phyTs":{pyId:[ts]},"params":{"nSpectrumType":0}}
        print(payload)
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 200:
            data = response.json()['data']['yVals']
            self.write_to_influxDB(fields,data)
        else:
            print(response.content)


    def write_to_influxDB(self,fields,data):
        idx = 0
        points = []
        write_api = self.client.write_api(write_options=SYNCHRONOUS)
        file_name  = fields['fileName']
        measurement = file_name.split("_")[0]
        node_name = file_name.split("#")[1].split(".")[0]
        trigger_at = file_name.split("_")[1].split("#")[0]
        start_time_ms = fields['ts']
        print(f"measurement = {measurement} | node_name = {node_name} | trigger_at = {trigger_at}")
        for val in data:
            point = (
                Point(measurement)
                .tag("trigger_at", trigger_at)
                .tag("unit", fields['eu'])
                .field(node_name, val)
                .time(start_time_ms)
            )
            points.append(point)
            idx  += 1

            if idx == int(fields['simple_freq'])  :

                start_time_ms +=  1_000  # 时间增加 1 秒，单位为毫秒
                idx = 0
                print("write")
                # write_api.write(bucket=self.bucket, org="g_lab", record=points)
                write_api.flush()
                points = []
            start_time_ms += 1  
            
        print("*"*120,len(data)) 
        if len(points)>0:
            # write_api.write(bucket=self.bucket, org="g_lab", record=points)
            print("write")
            write_api.flush()


    def request_point_list(self,bucket,pyId):
        response = requests.get(f'{self.base_url}api/ts/queryPyData?bucket={bucket}&pyKey=inv_file&pyId={pyId}&start=1601481600000&stop=1696953599000&keys=ts,ts,space,fileName,simple_freq,point_number,eu,file_tsval,file_fft,is_alarm&page=0&size=10&sort=_time,desc'
                                ,headers=self.headers)
        if response.status_code == 200:
            data = response.json()

            with open("mts.json","r",encoding="utf-8") as f:
                mts_dict = json.load(f)

            # 第一页数据文件
            if int(data['numberOfElements']) > 0:
                
                for content in data['content']:
                    
                    prefix_name = content['fields']['fileName'].split(".")[0]
                    if str(prefix_name+".tsp") in mts_dict:
                        print(f" file {prefix_name} continue.     ")
                        continue
                    self.down_data(content['pyId'],content['ts'],content['fields'])
                    return

            # 剩余数据文件
            for idx in range(int(data['totalPages'])):
                idx_url = f'{self.base_url}api/ts/queryPyData?bucket={bucket}&pyKey=inv_file&pyId={pyId}&start=1601481600000&stop=1696953599000&keys=ts,ts,space,fileName,simple_freq,point_number,eu,file_tsval,file_fft,is_alarm&page={idx + 1}&size=10&sort=_time,desc'
                print(idx_url)
                response = requests.get(idx_url,headers=self.headers)
                if response.status_code == 200:
                    child_data = response.json()
                    if int(child_data['numberOfElements']) > 0:
                        ts = []
                        for content in child_data['content']:
                            prefix_name = content['fields']['fileName'].split(".")[0]
                            if str(prefix_name+".tsp") in mts_dict:
                                print(f" file {prefix_name} continue.     ")
                                continue
                            ts.append(content['ts'])
                        if len(ts)>0:
                            self.parseBinary.parse_binary(analyzeId=self.analyzeId,pyId=content['pyId'],ts=ts,http_headers=self.headers)

    def request_pages(self):
        with open("points.json","r",encoding="utf-8") as f:
            exists_points = json.load(f)
        url = f'{self.base_url}api/analyzes/phyTreeData/{self.analyzeId}'
        print(url)
        response  =requests.get(url,headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            procs = []
            for item in data:
                if item['children'] :
                    for val in item['children']:
                        if val['children'] :
                            for last in val['children']: 
                                begin = time.time()
                                extra = last['extra']
                                if extra['title'] in exists_points : 
                                    print(f"{extra['title']} continue.")
                                    continue
                                # if (extra['title'] != "A12-梁体跨中动应变"):continue
                                print(f"bucket or analyzeId = {extra['tenantId']} | pyId = {extra['id']} title = {extra['title']}")
                                self.request_point_list(extra['tenantId'],extra['id'])

                                with open("points.json","w",encoding="utf-8") as f :
                                    print(extra['title'])
                                    exists_points[extra['title']] = extra
                                    # json.dump(exists_points,f)
                                print(f"single point cost time {begin - time.time()} s")
                                print("===="*40)
                                return

token = 'Bearer eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ6bmR4X2FkbWluQGNvaW52LmNvbSIsImF1dGgiOiJURU5BTlRfQURNSU4iLCJ0ZW5hbnRJZCI6IjYyN2NhMzhmZjEyZDk0NWZjZWJmNjk4YSIsImV4cCI6MTY5OTUyNDEzN30.sBJDvsTkwHrYTAiMR0G7F_VFp_vkxd6rIOiXtC4vyUsXzMIfDBKYnP78SLoQRZ6gU3bI-GWIui57xPgVYJK-nw'
# analyzeId = '65111765a06cdd5098f01285' 滁宁原始数据
analyzeId = '6361bb96a768a45b82629b72' # 朔黄重载铁路桥梁 - 自振特性原始数据
# analyzeId = '6361bbe9a768a45b82629b84' # 朔黄重载铁路桥梁 - 动力响应原始数据
file_mate = FileMate(analyzeId,token)
#save_binary_as_zip(save_path)
# file_mate.request_pages()
attempts = 0
max_attempts = 16
while attempts < max_attempts:
    try:
        file_mate.request_pages()
        break
    except Exception as e:
        print(f"Error: {e}")
        attempts += 1
        print(f"Retrying... (Attempt {attempts} of {max_attempts})")
        time.sleep(10)

