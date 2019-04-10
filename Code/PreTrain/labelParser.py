#coding=utf-8
import string
import xml.sax

class LabelParserHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.objectname = ""
        self.topleft = []
        self.bottomright = []
        self.objectlist = []
        self.object = {}
    def startElement(self,tag,attributes):
        self.CurrentTag = tag
    def endElement(self,tag):
        if tag == "object":
            self.objectlist.append((self.object.copy()))
            self.object.clear()
            self.objectname = ""
            self.bottomright.clear()
            self.topleft.clear()
        elif tag == "bndbox":
            self.object["name"] = (self.objectname)
            self.object["topleft"] = (tuple(self.topleft))
            self.object["bottomright"] = (tuple(self.bottomright))
    def characters(self,content):
        if ((content.isalnum())):
            if self.CurrentTag == "name":
                self.objectname += content
            elif self.CurrentTag == "xmin":
                self.topleft.append((content))
            elif self.CurrentTag == "ymin":
                self.topleft.append((content))
            elif self.CurrentTag == "xmax":
                self.bottomright.append((content))
            elif self.CurrentTag == 'ymax':
                self.bottomright.append((content))
    def labelresult(self):
        return self.objectlist