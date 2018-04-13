
import threading, os, utils
from collections import OrderedDict
data_folder = 'data'
last_update_time = None
info_folder = 'eval-results'
info_path = 'client/' + info_folder + '/info.txt'
info = None
lock = threading.Lock()

def scan_files(data_folder, extension='.tif'):
	import os
	files_scanned = []
	for root, dirs, files in os.walk(data_folder):
		if root != data_folder: continue
		for file in files:
			if file.endswith(extension) and not file.startswith('._'):
				files_scanned.append(file)
	return files_scanned

def reload_info():
	import copy
	global info_path, data_folder, lock, info, last_update_time
	if info is None:
		info = OrderedDict()
		if os.path.isfile(info_path):
			info = utils.load_json(info_path)
			for file in info:
				if info[file]['result'] and not isinstance(info[file]['result'], (tuple, list)):
					info[file]['result'] = None
			last_update_time = utils.cur_time_str()
	files = scan_files(data_folder)
	lock.acquire()
	old_info = copy.deepcopy(info)
	for file in info:
		info[file]['presents'] = False
	for file in files:
		if file not in info:
			info[file] = { 
						'presents':True,
						'fileName':file,
						'thumbnail':None,
						'modificationTime':utils.file_mtime(os.path.join(data_folder, file)),
						'result':None
						}
		else:
			info[file]['presents'] = True
	if info != old_info:
		info = OrderedDict(sorted(info.items(), key=lambda item: item[1]['modificationTime'][0], reverse=True))
		last_update_time = utils.cur_time_str()
		utils.save_json(info_path, info)
		print('Updated: ' + last_update_time)
	lock.release()

def reload_info_thread():
	import time
	while True:
		reload_info()
		time.sleep(1)

def getCurrentRecords():
	global info, lock
	records = []
	lock.acquire()
	for file in info:
		record = {}
		if not info[file]['presents']: continue
		if info[file]['thumbnail']:
			record['image'] = "<img src='%s' style='max-height:180px'/>" % info[file]['thumbnail']
		else:
			record['image'] = "<img src='img/processing.gif' />"
		record['label'] = "<h3>%s</h3><p>修改日期: %s</p>" % (info[file]['fileName'], info[file]['modificationTime'][1])
		if info[file]['result']:
			if isinstance(info[file]['result'], str):
				record['label'] += '''<div class="progress">
					<div class="progress-bar progress-bar-success progress-bar-striped" role="progressbar" aria-valuenow="%s" aria-valuemin="0" aria-valuemax="100" style="width:%s">
					%s
					</div>
				</div>''' % (info[file]['result'].split('%')[0], info[file]['result'], info[file]['result'])
			elif isinstance(info[file]['result'], (tuple, list)):
				record['label'] += '<p><a href="%s" class="btn btn-success" >分析结果</a></p>' % info[file]['result'][0]
		else:
			record['label'] += '''<div class="progress">正在等待分析
					<div class="progress-bar progress-bar-success progress-bar-striped" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%">
					</div>
				</div>'''
		record['label'] += '<a href="#" class="btn btn-danger" onclick="onButtonClick(\'/table/delete\', {\'file\':\'%s\'}, \'请确认删除切片文件及所有数据。\')" >删除切片文件及所有数据</a>' % file
		records.append(record)
	lock.release()
	return records

def get_thumbnail(slide_path, thumbnail_path):
	import openslide, cv2
	import numpy as np
	map_level = 6
	try:
		slide = openslide.OpenSlide(slide_path)
		slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[map_level]))
		cv2.imwrite(thumbnail_path, slide_map)
		return True
	except:
		return False

def make_thumbnail():
	global info, lock, last_update_time
	slide_file = None
	lock.acquire()
	for file in info:
		if not info[file]['presents']: continue
		if info[file]['thumbnail'] is None:
			slide_file = file
			break
	lock.release()
	if not slide_file: return False

	print('Making thumbnail for %s' % slide_file)
	thumbnail_file = os.path.join(info_folder, slide_file + '-thumbnail.png');
	if not get_thumbnail(os.path.join(data_folder, slide_file), os.path.join('client', thumbnail_file)):
		return False
	lock.acquire()
	info[slide_file]['thumbnail'] = thumbnail_file
	last_update_time = utils.cur_time_str()
	utils.save_json(info_path, info)
	print('Updated: ' + last_update_time)
	lock.release()
	return True

def thumbnail_thread():
	import time
	while True:
		if not make_thumbnail():
			time.sleep(1)

def predict_file():
	global info, lock, last_update_time
	slide_file = None
	lock.acquire()
	for file in info:
		if not info[file]['presents']: continue
		if info[file]['result'] is None or not isinstance(info[file]['result'], (tuple, list)): 
			slide_file = file
			break
	lock.release()
	if not slide_file: return False

	def update_progress(progress):
		global info, lock, last_update_time
		lock.acquire()
		progress_text = None
		if progress >= 0:
			progress_text = '%d%%' % int(progress)
		if info[slide_file]['result'] != progress_text:
			info[slide_file]['result'] = progress_text
			last_update_time = utils.cur_time_str()
			# utils.save_json(info_path, info)
			# print('Updated: ' + last_update_time)
		lock.release()

	import eval
	print('Evaluating %s' % slide_file)
	result_file = os.path.join(info_folder, slide_file + '-result.png');
	try:
		width, height = eval.eval(os.path.join(data_folder, slide_file), os.path.join('client', result_file), update_progress)
	except:
		update_progress(-1)
		return False
	html_path = os.path.join(info_folder, slide_file + '-result.html');

	def create_html(width, height, thumbnail_path, result_file, html_path):
		html = '<img  width=%d height=%d src=%s>' % (width, height, thumbnail_path.split('/')[-1])
		html += '<img  width=%d height=%d src=%s>' % (width, height, result_file.split('/')[-1])
		utils.save_file(os.path.join('client', html_path), html)
	create_html(width, height, info[slide_file]['thumbnail'], result_file, html_path)

	lock.acquire()
	info[slide_file]['result'] = (html_path, result_file)
	last_update_time = utils.cur_time_str()
	utils.save_json(info_path, info)
	print('Updated: ' + last_update_time)
	lock.release()
	return True	

def predict_thread():
	import time
	global cur_predict_file, cur_predict_precentage 
	while True:
		if not predict_file():
			time.sleep(1)

def serve():
	thread = threading.Thread(target=reload_info_thread)
	thread.setDaemon(True)
	thread.start()

	thread = threading.Thread(target=thumbnail_thread)
	thread.setDaemon(True)
	thread.start()

	thread = threading.Thread(target=predict_thread)
	thread.setDaemon(True)
	thread.start()

	# from pyramid.session import SignedCookieSessionFactory
	from pyramid.config import Configurator
	from waitress import serve

	# session_factory = SignedCookieSessionFactory('biocoding.cn.server_secret')

	# with Configurator(session_factory=session_factory) as config:
	with Configurator() as config:
		config.add_route('index', '/')
		config.add_route('table', '/table/{service_name}')
		config.add_static_view(name='client', path='../client')
		config.scan('.')
		app = config.make_wsgi_app()
		serve(app, host='0.0.0.0', port=8888)

from pyramid.view import (view_config, view_defaults)

@view_defaults(renderer='json')
class Views:

	def __init__(self, request):
		self.request = request

	@view_config(route_name='index')
	def home(self):
		from pyramid.httpexceptions import HTTPFound
		return HTTPFound(location='client/list.html') # redirection

	@view_config(route_name='table')
	def table(self):
		global last_update_time
		id_ = None
		try:
			service_name = self.request.matchdict['service_name']
			params = self.request.params
			# session = self.request.session
			if service_name == 'info':
				# print(str(datetime.datetime.now()))
				records = getCurrentRecords()
				reply = { 'lastUpdateTime':  last_update_time, 'records': records }
				return reply;
			if service_name == 'delete':
				file = params['file']
				return delete_file(file)
			return {'error': 'No such service: %s' % service_name}
		except Exception as e:
			import traceback
			traceback.print_exc()
			return {'error': str(e)}


def delete_file(file):
	global lock, info, last_update_time
	path = os.path.join(data_folder, file)
	if os.path.isfile(path):
		try:
			print('Removing file %s' % path)
			os.remove(path)
		except FileNotFoundError:
			return { 'error': '不能删除文件 %s，请稍后再尝试。' % file }
		thumbnail_file = None
		lock.acquire()
		thumbnail_file = info[file]['thumbnail']
		result = info[file]['result']
		del info[file]
		last_update_time = utils.cur_time_str()
		utils.save_json(info_path, info)
		lock.release()
		if thumbnail_file:
			try:
				path = os.path.join('client', thumbnail_file)
				print('Removing file %s' % path)
				os.remove(path)
			except FileNotFoundError:
				pass
		if result and isinstance(result, (tuple, list)):
			for result1 in result:
				try:
					path = os.path.join('client', result1)
					print('Removing file %s' % path)
					os.remove(path)
				except FileNotFoundError:
					pass
		return { 'reply': 'OK' }
	else:
		return { 'error': '文件 %s 不存在。' % file }

def datetimestr():
	def get_text(value):
		if value < 10:
			return '0%d' % value
		else:
			return '%d' % value
	from datetime import datetime
	now = datetime.now()
	return get_text(now.year) + get_text(now.month) + get_text(now.day) + '_' + \
			get_text(now.hour) + get_text(now.minute) + get_text(now.second)

