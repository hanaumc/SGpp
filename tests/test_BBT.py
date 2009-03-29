#############################################################################
# This file is part of pysgpp, a program package making use of spatially    #
# adaptive sparse grids to solve numerical problems                         #
#                                                                           #
# Copyright (C) 2009 Alexander Heinecke (Alexander.Heinecke@mytum.de)       #
#                                                                           #
# pysgpp is free software; you can redistribute it and/or modify            #
# it under the terms of the GNU Lesser Public License as published by       #
# the Free Software Foundation; either version 3 of the License, or         #
# (at your option) any later version.                                       #
#                                                                           #
# pysgpp is distributed in the hope that it will be useful,                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of            #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             #
# GNU Lesser General Public License for more details.                       #
#                                                                           #
# You should have received a copy of the GNU Lesser Public License          #
# along with pysgpp; if not, write to the Free Software                     #
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA #
# or see <http://www.gnu.org/licenses/>.                                    #
#############################################################################

import unittest, tools

#-------------------------------------------------------------------------------
## Builds the training data vector
# 
# @param data a list of lists that contains the points a the training data set, coordinate-wise
# @return a instance of a DataVector that stores the training data
def buildTrainingVector(data):
    from pysgpp import DataVector
    dim = len(data["data"])
    training = DataVector(len(data["data"][0]), dim)
    
    # i iterates over the data points, d over the dimension of one data point
    for i in xrange(len(data["data"][0])):
        for d in xrange(dim):
            training[i*dim + d] = data["data"][d][i]
    
    return training


def openFile(filename):
    try:
        data = tools.readDataARFF(filename)
    except:
        print ("An error occured while reading " + filename + "!")
        
    if data.has_key("classes") == False:
        print ("No classes found in the given File " + filename + "!")
        
    return data


def generateBBTMatrix(factory, training, verbose=False):
    from pysgpp import DataVector
    storage = factory.getStorage()
       
    b = factory.createOperationB()
    
    alpha = DataVector(storage.size())
    erg = DataVector(alpha.getSize())
    temp = DataVector(training.getSize())
    
    # create B matrix
    m = DataVector(storage.size(), storage.size())
    for i in xrange(storage.size()):
        # apply unit vectors
        temp.setAll(0.0)
        erg.setAll(0.0)
        alpha.setAll(0.0)
        alpha[i] = 1.0
        b.multTranspose(alpha, training, temp)
        b.mult(temp, training, erg)
        #Sets the column in m
        m.setColumn(i, erg)
        
    return m


def readReferenceMatrix(self, storage, filename):
    from pysgpp import DataVector
    # read reference matrix
    try:
        fd = tools.gzOpen(filename, 'r')
    except IOError, e:
        fd = None
        
    if not fd:
        fd = tools.gzOpen('tests/' + filename, 'r')
        
    dat = fd.read().strip()
    fd.close()
    dat = dat.split('\n')
    dat = map(lambda l: l.strip().split(None), dat)

    # right number of entries?
    self.assertEqual(storage.size(), len(dat))
    self.assertEqual(storage.size(), len(dat[0]))

    m_ref = DataVector(len(dat), len(dat[0]))
    for i in xrange(len(dat)):
        for j in xrange(len(dat[0])):
            m_ref[i*len(dat) + j] = float(dat[i][j])

    return m_ref

def readDataVector(filename):
    from pysgpp import DataVector
    
    try:
        fin = tools.gzOpen(filename, 'r')
    except IOError, e:
        fin = None
        
    if not fin:
        fin = tools.gzOpen('tests/' + filename, 'r')
    
    data = []
    classes = []
    hasclass = False

    # get the different section of ARFF-File
    for line in fin:
        sline = line.strip().lower()
        if sline.startswith("%") or len(sline) == 0:
            continue

        if sline.startswith("@data"):
            break
        
        if sline.startswith("@attribute"):
            value = sline.split()
            if value[1].startswith("class"):
                hasclass = True
            else:
                data.append([])
    
    #read in the data stored in the ARFF file
    for line in fin:
        sline = line.strip()
        if sline.startswith("%") or len(sline) == 0:
            continue

        values = sline.split(",")
        if hasclass:
            classes.append(float(values[-1]))
            values = values[:-1]
        for i in xrange(len(values)):
            data[i].append(float(values[i]))
            
    # cleaning up and return
    fin.close()
    return {"data":data, "classes":classes, "filename":filename}

##
# Compares, if two matrices are "almost" equal.
# Has to handle the problem that the underlying grid was ordered
# differently. Uses heuristics, e.g. whether the diagonal elements
# and row and column sums match.
def compareBBTMatrices(testCaseClass, m1, m2):
    from pysgpp import DataVector

    # check dimensions
    testCaseClass.assertEqual(m1.getSize(), m1.getDim())
    testCaseClass.assertEqual(m1.getSize(), m2.getSize())
    testCaseClass.assertEqual(m1.getDim(), m2.getDim())

    n = m1.getSize()

    # check diagonal
    values = []
    for i in range(n):
        values.append(m1[i*n + i])
    values.sort()
    values_ref = []
    for i in range(n):
        values_ref.append(m2[i*n + i])
    values_ref.sort()
    for i in range(n):
        testCaseClass.assertAlmostEqual(values[i], values_ref[i], msg="Diagonal %f != %f" % (values[i], values_ref[i]))

    # check row sum
    v = DataVector(n)
    values = []
    for i in range(n):
        m1.getRow(i,v)
        values.append(v.sum())
    values.sort()
    values_ref = []
    for i in range(n):
        m2.getRow(i,v)
        values_ref.append(v.sum())
    values_ref.sort()
    for i in range(n):
        #print values_ref[i], values[i]
        testCaseClass.assertAlmostEqual(values[i], values_ref[i], msg="Row sum %f != %f" % (values[i], values_ref[i]))

    # check col sum
    v = DataVector(n)
    values = []
    for i in range(n):
        m1.getColumn(i,v)
        values.append(v.sum())
    values.sort()
    values_ref = []
    for i in range(n):
        m2.getColumn(i,v)
        values_ref.append(v.sum())
    values_ref.sort()
    for i in range(n):
        testCaseClass.assertAlmostEqual(values[i], values_ref[i], msg="Col sum %f != %f" % (values[i], values_ref[i]))



class TestOperationBBTModLinear(unittest.TestCase):
    ##
    # Test laplace for regular sparse grid in 1d using linear hat functions
    def testHatRegular1D_one(self):
        from pysgpp import Grid
        
        factory = Grid.createModLinearGrid(1)
        training = buildTrainingVector(readDataVector('data/data_dim_1_nops_8_float.arff.gz'))
        level = 3
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_ausgeklappt_dim_1_nopsgrid_7_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 

  
    ##
    # Test laplace for regular sparse grid in 1d using linear hat functions
    def testHatRegular1D_two(self):
        from pysgpp import Grid
        
        factory = Grid.createModLinearGrid(1)
        training = buildTrainingVector(readDataVector('data/data_dim_1_nops_8_float.arff.gz'))
        level = 5
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_ausgeklappt_dim_1_nopsgrid_31_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 

                
    ##
    # Test regular sparse grid dD, normal hat basis functions.
    def testHatRegulardD_one(self):  
        from pysgpp import Grid
        
        factory = Grid.createModLinearGrid(3)
        training = buildTrainingVector(readDataVector('data/data_dim_3_nops_512_float.arff.gz'))
        level = 3
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_ausgeklappt_dim_3_nopsgrid_31_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 
  
        
    ##
    # Test regular sparse grid dD, normal hat basis functions.
    def testHatRegulardD_two(self):
        from pysgpp import Grid
        
        factory = Grid.createModLinearGrid(3)
        training = buildTrainingVector(readDataVector('data/data_dim_3_nops_512_float.arff.gz'))
        level = 4
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_ausgeklappt_dim_3_nopsgrid_111_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 
             

class TestOperationBBTLinear(unittest.TestCase):
    ##
    # Test laplace for regular sparse grid in 1d using linear hat functions
    def testHatRegular1D_one(self):
        from pysgpp import Grid
        
        factory = Grid.createLinearGrid(1)
        training = buildTrainingVector(readDataVector('data/data_dim_1_nops_8_float.arff.gz'))
        level = 3
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_dim_1_nopsgrid_7_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 

  
    ##
    # Test laplace for regular sparse grid in 1d using linear hat functions
    def testHatRegular1D_two(self):
        from pysgpp import Grid
        
        factory = Grid.createLinearGrid(1)
        training = buildTrainingVector(readDataVector('data/data_dim_1_nops_8_float.arff.gz'))
        level = 5
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_dim_1_nopsgrid_31_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 

                
    ##
    # Test regular sparse grid dD, normal hat basis functions.
    def testHatRegulardD_one(self):  
        from pysgpp import Grid
        
        factory = Grid.createLinearGrid(3)
        training = buildTrainingVector(readDataVector('data/data_dim_3_nops_512_float.arff.gz'))
        level = 3
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_dim_3_nopsgrid_31_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 
  
        
    ##
    # Test regular sparse grid dD, normal hat basis functions.
    def testHatRegulardD_two(self):
        from pysgpp import Grid
        
        factory = Grid.createLinearGrid(3)
        training = buildTrainingVector(readDataVector('data/data_dim_3_nops_512_float.arff.gz'))
        level = 4
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_dim_3_nopsgrid_111_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 
             

class TestOperationBBTLinearBoundary(unittest.TestCase):
    ##
    # Test laplace for regular sparse grid in 1d using linear hat functions
    def testHatRegular1D_one(self):
        from pysgpp import Grid
        
        factory = Grid.createLinearBoundaryGrid(1)
        training = buildTrainingVector(readDataVector('data/data_dim_1_nops_8_float.arff.gz'))
        level = 4
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_l0_rand_dim_1_nopsgrid_17_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 

  
    ##
    # Test laplace for regular sparse grid in 1d using linear hat functions
    def testHatRegular1D_two(self):
        from pysgpp import Grid
        
        factory = Grid.createLinearBoundaryGrid(1)
        training = buildTrainingVector(readDataVector('data/data_dim_1_nops_8_float.arff.gz'))
        level = 5
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_l0_rand_dim_1_nopsgrid_33_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 

                
    ##
    # Test regular sparse grid dD, normal hat basis functions.
    def testHatRegulardD_one(self):  
        from pysgpp import Grid
        
        factory = Grid.createLinearBoundaryGrid(3)
        training = buildTrainingVector(readDataVector('data/data_dim_3_nops_512_float.arff.gz'))
        level = 3
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_l0_rand_dim_3_nopsgrid_123_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 
  
        
    ##
    # Test regular sparse grid dD, normal hat basis functions.
    def testHatRegulardD_two(self):
        from pysgpp import Grid
        
        factory = Grid.createLinearBoundaryGrid(3)
        training = buildTrainingVector(readDataVector('data/data_dim_3_nops_512_float.arff.gz'))
        level = 4
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_l0_rand_dim_3_nopsgrid_297_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref)     
        

class TestOperationBBTLinearBoundaryUscaled(unittest.TestCase):
    ##
    # Test laplace for regular sparse grid in 1d using linear hat functions
    def testHatRegular1D_one(self):
        from pysgpp import Grid
        
        factory = Grid.createLinearBoundaryUScaledGrid(1)
        training = buildTrainingVector(readDataVector('data/data_dim_1_nops_8_float.arff.gz'))
        level = 4
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_trapezrand_dim_1_nopsgrid_17_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 

  
    ##
    # Test laplace for regular sparse grid in 1d using linear hat functions
    def testHatRegular1D_two(self):
        from pysgpp import Grid
        
        factory = Grid.createLinearBoundaryUScaledGrid(1)
        training = buildTrainingVector(readDataVector('data/data_dim_1_nops_8_float.arff.gz'))
        level = 5
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_trapezrand_dim_1_nopsgrid_33_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 

                
    ##
    # Test regular sparse grid dD, normal hat basis functions.
    def testHatRegulardD_one(self):  
        from pysgpp import Grid
        
        factory = Grid.createLinearBoundaryUScaledGrid(3)
        training = buildTrainingVector(readDataVector('data/data_dim_3_nops_512_float.arff.gz'))
        level = 2
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_trapezrand_dim_3_nopsgrid_81_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref) 
  
        
    ##
    # Test regular sparse grid dD, normal hat basis functions.
    def testHatRegulardD_two(self):
        from pysgpp import Grid
        
        factory = Grid.createLinearBoundaryUScaledGrid(3)
        training = buildTrainingVector(readDataVector('data/data_dim_3_nops_512_float.arff.gz'))
        level = 3
        gen = factory.createGridGenerator()
        gen.regular(level)

        m = generateBBTMatrix(factory, training)
        m_ref = readReferenceMatrix(self, factory.getStorage(), 'data/BBT_phi_li_hut_trapezrand_dim_3_nopsgrid_225_float.dat.gz')

        # compare
        compareBBTMatrices(self, m, m_ref)  
        
                                       
# Run tests for this file if executed as application 
if __name__=='__main__':
    unittest.main()

