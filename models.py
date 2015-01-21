from os.path import *
import os
import re
import csv

# Model for general MR image subject
class mrimg (object):

    def __init__(self, meta={}, filepath=""):
        self._meta = meta # The meta information of each image is dynamic
        self._imgid = self.getimgid()
        self._filepath = filepath # The file path is optional here, 
                                  # since the nipype dataGrabber will find it out according to the Image ID

    def findflirtedimg(self, flirted):
        imgid = self._getimgid()
        foundfile = [ f for f in os.listdir(path) if os.path.isfile(join(path,f)) and f.endswith('.nii.gz') and '_I'+imgid in f]

        if len(foundfile) == 0:
          raise Exception('No flirt image associated with %s ' % imgid)
        elif len(foundfile) > 1:
          raise Exception('%d Duplicated image ID found for %s ' % (len(foundfile), imgid))
        return join(path, foundfile[0])

    def getmetafield(self, fieldname):
        return self._meta[fieldname]

    def getfilepath(self):
        return abspath(self._filepath)


# Model for ADNI MR image which defined by the merged csv table
class adnimrimg (mrimg):

    def __init__(self, meta={}, filepath=""):
        mrimg.__init__(self, meta, filepath)

    def getimgid(self):
        return self._meta['Image.Data.ID']

    def getviscode(self):
        return int(self._meta['VISCODE'].replace('m', ''))


class transformpair (object):

    def __init__(self, fixedimage=None, movingimage=None):
        self.fixedimage = fixedimage 
        self.movingimage = movingimage

    def getinterval(self):
        return self.fixedimage.getviscode() - self.movingimage.getviscode()


# A collection tool class to perform some grouping and searching of the adnimriimg models
class AdniMrCollection(object):

    def __init__(self, ladnimr=None, dbpath=None):
        if ladnimr is not None:
            self._ladnimr = ladnimr 
        elif dbpath is not None:
            self._ladnimr = self.build_adni_mrlist(dbpath)


    def build_adni_mrlist(self, dbpath):
        # Find all images stated in the dbgen.csv in the db path
        # Construct adnimrimg list
        limg = self._traverse_for_file(dbpath, '.nii')
        limgid = [self._find_adni_imgid(img[1]) for img in limg ]
        self._ladnimr = []

        # read in the dbgen.db
        with open(join(dbpath, 'dbgen.csv')) as f:
          r = csv.reader(f, delimiter=',')
          header = next(r)
          #imgididx = header.index('Image.Data.ID')

          for row in r:
            meta = {}
            for i, col in enumerate(row):
                meta[header[i]] = col

            imgid = meta['Image.Data.ID']
            '''
            print('=====limgid==============')
            print(limgid) 
            print('===================')
            '''

            imgf = limg[ limgid.index(imgid) ]
            filepath = join(imgf[0], imgf[1])
            self._ladnimr.append(adnimrimg(meta, filepath))

        return self._ladnimr


    def getmrlist(self):
        return self._ladnimr


    # Return Image.Data.ID pairs [[fixed_img_id1, moving_img_id1], [fixed_img_id2, moving_img_id2],...]
    def find_transform_pairs(self, interval=[6,12]):
        sbjdict = self.group_sbj()
        pairs = []

        # Find the intervals with the specific lengths
        for key, sbj_imglist in sbjdict.iteritems():
            viscode_sortlist  = sorted(sbj_imglist, key=lambda x: x.getmetafield('VISCODE'))

            for i in xrange(len(viscode_sortlist)):
                for j in xrange(i+1,len(viscode_sortlist)):
                    viscode1 = viscode_sortlist[i].getviscode()
                    viscode2 = viscode_sortlist[j].getviscode()
                    if (viscode2 - viscode1) in interval:
                        pairs.append(transformpair(viscode_sortlist[j], viscode_sortlist[i]))

        return pairs


    # Group models into an RID dictionary <RID, [(VISCODE1, Image.Data.ID1), ...]> 
    def group_sbj(self):
        sbjdict = {}
        for model in self._ladnimr:
            rid     = model.getmetafield('RID') 

            if rid in sbjdict:
                sbjdict[rid].append(model)
            else:
                sbjdict[rid] = [model]

        return sbjdict


    def group_pairs(self, pairs=None, interval=[6,12]):
        if pairs == None:
            pairs = self.find_transform_pairs(interval)

        # Group pairs
        sbjdict = {}
        for pair in pairs:
            rid = pair.fixedimage.getmetafield('RID')

            if rid in sbjdict:
                sbjdict[rid].append(pair)
            else:
                sbjdict[rid] = [pair]

        return sbjdict


    def filter_elligible_pairs(self, pairs=None, interval=[6,12]):
        sbjdict = self.group_pairs(pairs, interval)

        elligible_pairs = []

        for sbjid, translist in sbjdict.iteritems():
            if len(translist) < 3: 
                continue

            l_moving_viscode = [t.movingimage.getviscode() for t in translist]

            for trans in translist:
                # see viscode 2 has a matching viscode1 means it has at list one follow up transforms
                if trans.fixedimage.getviscode() in l_moving_viscode:
                    elligible_pairs.append(trans)

        return elligible_pairs
    

    def find_followups(self, pairs, interval):
        search_sbjdict = self.group_pairs(pairs, interval)
        allpairs = find_transform_pairs(self, interval) # Find all pairs
        all_sbjdict = self.group_pairs(allpairs, interval)
        followupdict = {}

        for sbjid, translist in sbjdict.iteritems():
            for target in translist:
                allsbjlist = all_sbjdict[sbjid]
                followup = [p for p in allsbjlist 
                                  if p.movingimage.getviscode() == target.fixedimage.getviscode()
                                     and p.getinterval() in interval]
                if len(followup) == 0:
                    raise Exception('0 followup found for RID %s VISCODE %d-%d' % (target.fixedimage.getmetafield('RID'), 
                                                                                   target.movingimage.getviscode(),
                                                                                   target.fixedimage.getviscode()))
                followupdict[target] = followup
        
        return [followupdict[x] for x in pairs]

    # Traverse the dbpath for the files with provided suffix
    def _traverse_for_file(self, path, suffix):
        limg = [];
        for root, dirs, files in os.walk(path):
          for file in files:
            if file.endswith(suffix):
                limg.append((root, file))
        return limg

    # Find Image ID from ADNI nii filename
    def _find_adni_imgid(self, fname):
        lmatch = re.findall('_I\d+', fname)
        assert len(lmatch) <= 1 , 'More than one matches were found: '
        cleanmatch = []

        for m in lmatch:
          cleanmatch.append(m[2:]) # Remove the '_' before image id 

        return cleanmatch[0] if len(cleanmatch) == 1 else None