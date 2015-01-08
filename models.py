from os.path import abspath

# Model for general MR image subject
class mrimg (object):
    def __init__(self, meta={}, filepath=""):
        self._meta = meta # The meta information of each image is dynamic
        self._imgid = self.get_imgid()
        self._filepath = filepath 

    def findflirtedimg(self, flirted):
        imgid = self._get_imgid()
        foundfile = [ f for f in os.listdir(path) if os.path.isfile(join(path,f)) and f.endswith('.nii.gz') and '_I'+imgid in f]

        if len(foundfile) == 0:
          raise Exception('No flirt image associated with %s ' % imgid)
        elif len(foundfile) > 1:
          raise Exception('%d Duplicated image ID found for %s ' % (len(foundfile), imgid))
        return join(path, foundfile[0])

    def get_imgid(self):
        return self._meta['imgid'] # assumed name of this field

    def getmetafield(self, fieldname):
        return self._meta[fieldname]

    def getfilepath(self):
        return abspath(self._filepath)


# Model for ADNI MR image which defined by the merged csv table
class adnimrimg (mrimg):
    def __init__(self, meta={}, filepath=""):
        mrimg.__init__(self, meta, filepath)

    def get_imgid(self):
        return self._meta['Image.Data.ID']