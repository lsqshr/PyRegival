from utils import *
import numpy as np

class MrPredictor(object):
	def __init__(self, dbpath, ptemplate):
		self.dbpath = dbpath
		self.ptemplate = ptemplate

	def predict(self, mrpair, N, roweight=0.5, colweight=0.5, real_followup=None, option='change'):
		'''
		mrpair: tuple (mrid1, mrid2, interval)
		N : Int the number of neighbours to merge 
		roweight: the relative weight of the row elements for neighbood building
		colweight: the relative weight of the column elements for neighbood building
		option: 'change'/'baseline image'/'both'
		'''
		# Convert the similarity dict to a matrix with the order of the mrid pairs
		simmatrix, ind = self._convert_ptemplate2matrix() 

		# TODO: If this subject is not in the template, add this subject to the template

		# Find the column and row of this subject, 
		ind = elligible_pairs.index(mrpair)
		matrow = simmat[ind, :]
		matcol = simmat[:, ind]

		# calculate the row&column weights distribution considering the interval
		

		# Find the top N neighbours from row/column by weighting

		# Merge these templates by weighting

		# warp the target second image

		# Calculate the correlation with the real follow up


    def convert_ptemplate2matrix(self):
    	pairs = self.ptemplate['transpairs']
    	sim = self.ptemplate['corr']

		# Remove the transpairs without following transpairs
		elligible_pairs = AdniMrCollection(self.ptemplate['lmodel']).\
						                  filter_no_followup_pairs(pairs)
       	simmat = np.zeros((len(elligible_pairs), len(elligible_pairs)))

        for i, p1 in enumerate(elligible_pairs):
	        for j, p2 in enumerate(elligible_pairs):
	            simmat[i,j] = sim[(p1,p2)]

	    return simmat, elligible_pairs



