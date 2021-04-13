import os
import re
from glob import glob
import hwp5

class TextConverter():
	'''
	A class used to convert a file to a plain text file.

	Methods
	-------
	hwp2txt
		| Converts a HWP file to a plain text file. Returns 0 if no error occurs.
	'''
	def __init__(self):
		pass


	def hwp2txt(self, hwp_fpath, output_fpath):
		'''
		A method to convert a HWP file to a plain text file.

		Attributes
		----------
		hwp_fpath : str
			| The file path of HWP file to be converted.
		output_fpath : str
			| The path of file that contains plain text extracted from hwp_fpath.
		'''
		self.hwp_fpath = hwp_fpath
		self.output_fpath = output_fpath

		# TODO: the below block will be updated to use util.makedir()
		output_fdir = '/'.join(self.output_fpath.split('/')[:-1])
		os.makedirs(output_fdir, exist_ok=True)
		
		self.com_hwp = '"' + self.hwp_fpath + '"'
		self.com_output = '"' + self.output_fpath + '"'

		command = ' '.join(['hwp5txt', '--output', self.com_output, self.com_hwp])
		os.system(command)

		return 0
