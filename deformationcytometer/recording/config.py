import sys
import json
from configparser import ConfigParser


class SetupConfig():
	def __init__(self, confingPath):
		self.con = ConfigParser()
		self.con.read(confingPath)
		self.path = confingPath

		section = self.con['SETUP']
		self.pressure = int(section['pressure'].split()[0])
		self.imPos = section['imaging position after inlet']
		self.roomT = int(section['room temperature'].split()[0])
		self.bioink = section['bioink']
		self.duration = float(section['duration'].split()[0])

		section = self.con['MICROSCOPE']
		self.aperture = (section['condensor aperture'])
		self.obj = (section['objective'].split()[0])
		self.na = (section['na'])
		self.coupler = (section['coupler'].split()[0])

		section = self.con['CELL']
		self.Ctype = (section['cell type'])
		self.cellPnum = (section['cell passage number'])
		self.Tharvest = (section['time after harvest'])
		self.treatment = (section['treatment'])

		section = self.con['CAMERA']
		self.brsn = section['brightfield SN']
		self.flsn = section['fluorescent SN']
		self.frate = int(section['frame rate'].split()[0])
		self.exTrig = json.loads(section['external trigger'].lower())

		section = self.con['FLUORESCENT CHANNEL']
		self.flexp = int(section['exposure time'].split()[0])
		self.flgain = int(section['gain'])
		self.flXflip = json.loads(section['flip_x'].lower())
		self.flYflip = json.loads(section['flip_y'].lower())

		section = self.con['BRIGHTFIELD CHANNEL']
		self.brexp = int(section['exposure time'].split()[0])
		self.brgain = int(section['gain'])
		self.brXflip = json.loads(section['flip_x'].lower())
		self.brYflip = json.loads(section['flip_y'].lower())
		




	def update(self, main):
		main.frate.setValue(self.frate)

		main.roomtemp.setValue(self.roomT)
		main.obj.setText(self.obj)
		main.coupler.setText(self.coupler)
		main.NA.setText(self.na)

		main.pressure.setValue(self.pressure)
		main.impos.setText(self.imPos)
		main.aperture.setText(self.aperture)
		main.bioink.setText(self.bioink)
		main.celltype.setText(self.Ctype)
		main.cellpnum.setText(self.cellPnum)
		main.tharvest.setText(self.Tharvest)
		main.treatment.setText(self.treatment)
		main.duration.setValue(self.duration)

		main.flgainspin.setValue(self.flgain)
		main.flexpspin.setValue(self.flexp)
		main.brgainspin.setValue(self.brgain)
		main.brexpspin.setValue(self.brexp)

		main.brXflip = self.brXflip
		main.brYflip = self.brYflip
		main.flXflip = self.flXflip
		main.flYflip = self.flYflip
		
		for i in range(main.brsn.count()):
			if main.brsn.itemText(i) == self.brsn:
				main.brsn.setCurrentIndex(i)
			if main.brsn.itemText(i) == self.flsn:
				main.flsn.setCurrentIndex(i)

	def save(self , main):
		self.frate = main.frate.value()

		self.roomT = main.roomtemp.value()
		self.obj = main.obj.text()
		self.coupler = main.coupler.text()
		self.na = main.NA.text()

		self.pressure = main.pressure.value()
		self.imPos = main.impos.text()
		self.aperture = main.aperture.text()
		self.bioink = main.bioink.text()
		self.Ctype = main.celltype.text()
		self.cellPnum = main.cellpnum.text()
		self.Tharvest = main.tharvest.text()
		self.treatment = main.treatment.text()
		self.duration = main.duration.value()

		self.flgain = main.flgainspin.value()
		self.flexp = main.flexpspin.value()
		self.brgain = main.brgainspin.value()
		self.brexp = main.brexpspin.value()

		self.brsn = main.brsn.currentText()
		self.flsn = main.flsn.currentText()



		self.con['SETUP']['pressure'] = str(self.pressure) + ' kPa'
		self.con['SETUP']['imaging position after inlet'] = self.imPos
		self.con['SETUP']['room temperature'] = str(self.roomT) + ' C'
		self.con['SETUP']['bioink'] = self.bioink
		self.con['SETUP']['duration'] = str(self.duration) + ' s'
		self.con['MICROSCOPE']['condensor aperture'] = self.aperture
		self.con['MICROSCOPE']['objective'] = self.obj + ' x'
		self.con['MICROSCOPE']['na'] = self.na
		self.con['MICROSCOPE']['coupler'] = self.coupler + ' x'
		self.con['CELL']['cell type'] = self.Ctype
		self.con['CELL']['cell passage number'] = self.cellPnum
		self.con['CELL']['time after harvest'] = self.Tharvest
		self.con['CELL']['treatment'] = self.treatment
		self.con['FLUORESCENT CHANNEL']['gain'] = str(self.flgain)
		self.con['FLUORESCENT CHANNEL']['exposure time'] = str(self.flexp) + ' us' 
		self.con['BRIGHTFIELD CHANNEL']['gain'] = str(self.brgain)
		self.con['BRIGHTFIELD CHANNEL']['exposure time'] = str(self.brexp) + ' us' 

		self.con['CAMERA']['brightfield SN'] = str(self.brsn)
		self.con['CAMERA']['fluorescent SN'] = str(self.flsn)
		self.con['CAMERA']['frame rate'] = str(self.frate) + ' fps'
		self.con['CAMERA']['external trigger'] = str(self.exTrig) 

		# self.con.write(self.path)
		with open(self.path, 'w') as configfile:
			self.con.write(configfile)

	def savein(self , path):
		with open(path, 'w') as configfile:
			self.con.write(configfile)	


# def getFromConfig(config )

# for section in con:
# 	print(section)



# con = SetupConfig('config.txt')
# print(con.notes)



