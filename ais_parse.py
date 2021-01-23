from xml.dom import minidom as md
from datetime import datetime

class Vessel:
    def __init__(self, name, long, lat, time):
        self.name = name
        self.long = long
        self.lat = lat
        self.report_time = int(time)

    def __str__(self):
        return "Vessel | Name: " + self.name + "; Pos: (" + self.long + "," + self.lat + "); Last Reported: " + datetime.utcfromtimestamp(self.report_time).strftime('%Y-%m-%d %H:%M:%S')


def get_ais_info():
    xmldoc = md.parse('sample_response.xml')
    item_list = xmldoc.getElementsByTagName('vessel')
    vessel_list = []

    for v in item_list:
        v_name = v.attributes['NAME'].value
        v_long = v.attributes['LONGITUDE'].value
        v_lat = v.attributes['LATITUDE'].value
        v_report_time = v.attributes['TIME'].value
        vobj = Vessel(v_name, v_long, v_lat, v_report_time)
        vessel_list.append(vobj)
        print(vobj)

    return vessel_list
