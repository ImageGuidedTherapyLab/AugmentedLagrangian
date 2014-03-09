import vtk
import os
# echo vtk version info
print "using vtk version", vtk.vtkVersion.GetVTKVersion()

####################################################################
def ConvertGadgetronVTK(input_filename,output_filename):
  """
  http://sourceforge.net/p/gadgetron/home/Simple%20Array%20File%20Format/
  
  When working with the Gadgetron it is often necessary to write files with
  reconstructed images to disk, either as part of debugging or as the final
  reconstruction result. We have adopted a very simple multidimensional array
  file format for this purpose. The main advantage of this file format is its
  simplicity but there are a number of disadvantages and caveats as well as
  described in this section.
  
  The simple array files are made up of a) a header followed by b) the data
  itself. This layout of data and header is illustrated below. The header has a
  single 32-bit integer to indicate the number of dimensions of the dataset
  followed by one integer for each dimension to indicate the length of that
  dimension. The data follows immediately after the header. The data is stored
  such that the first dimension is the fastest moving dimension, second dimension
  is second fastest, etc. The header contains no information about the size of
  each individual data element and consequently the user needs to know what type
  of data is contained in the array. In general, the Gadgetron uses 3 different
  types of data and the convention is to use the file extension to indicate the
  data type in the file:
  
     16-bit unsigned short. File extension: *.short
     32-bit float. File extension: *.real
     32-bit complex float. Two 32-bit floating point values per data element. File extension: *.cplx
  
  The Gadgetron framework provides function for reading these files in C++. The
  functions are located in toolboxes/ndarray/hoNDArray_fileio.h in the Gadgetron
  source code distribution.
  
  It is also trivial to read the files into Matlab. Below is a function which
  detects the data type based on the file extension and reads the file into
  Matlab.
  
  """
  import vtk.util.numpy_support as vtkNumPy 
  import numpy
  import scipy.io as scipyio
 
  # read the header
  imagedimension = numpy.fromfile(input_filename, dtype=numpy.int32, count=1, sep='')[0]
  fileheader = numpy.fromfile(input_filename, dtype=numpy.int32, count=1+imagedimension, sep='')
  if( imagedimension == 1):
   dims = (fileheader[1],1,1)
  elif( imagedimension == 2):
   dims = (fileheader[1],fileheader[2],1)
  elif( imagedimension == 3):
   dims = (fileheader[1],fileheader[2],fileheader[3])
  else:
   raise RuntimeError('unknown dimension %d ' % imagedimension )

  # the extension is the datatype
  extension = input_filename.split('.').pop()
  dataImporter = vtk.vtkImageImport()
  if extension == 'short':
    datatype  = numpy.short
    dataImporter.SetDataScalarTypeToShort() 
  elif extension == 'real':
    datatype  = numpy.float32
    dataImporter.SetDataScalarTypeToFloat() 
  elif extension == 'cplx':
    datatype  = numpy.complex64
    dataImporter.SetDataScalarTypeToComplex() 
  else:
    raise RuntimeError('unknown data type %s ' % extension )

  # offset the data read by the header
  datafile = open(input_filename, "rb")  # reopen the file
  headeroffset = len(fileheader)*4       # 4 bytes per integer
  datafile.seek(headeroffset, os.SEEK_SET)  # seek
  numpy_data = numpy.fromfile(datafile, dtype=datatype, sep='')

  # error check
  ExpectedImageSize = 1
  for pixelsize in dims:
    ExpectedImageSize = ExpectedImageSize * pixelsize
  if( ExpectedImageSize != numpy_data.size):
    raise RuntimeError('file read error: expected size %d, size found %d ' % (ExpectedImageSize,numpy_data.size) )
    
  # convert to vtk
  spacing = [1.,1.,1.]
  dataImporter.SetNumberOfScalarComponents(1)
  dataImporter.SetDataExtent( 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
  dataImporter.SetWholeExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
  dataImporter.SetDataSpacing(spacing[0],spacing[1],spacing[2])
  numpy_data= numpy_data.reshape(dims[0],dims[1],dims[2])
  numpy_data= numpy_data.transpose(1,0,2)
  data_string = numpy_data.tostring()
  dataImporter.CopyImportVoidPointer(data_string, len(data_string))

  # write vtk file
  print "writing ", output_filename
  vtkImageDataWriter = vtk.vtkDataSetWriter()
  vtkImageDataWriter.SetFileTypeToBinary()
  vtkImageDataWriter.SetFileName( output_filename )
  vtkImageDataWriter.SetInput(dataImporter.GetOutput())
  vtkImageDataWriter.Update()

  #vtkReader.SetFileName( "%s" % (input_filename) ) 
  #vtkReader.Update()
  #imageDataVTK = vtkReader.GetOutput()
  #dimensions = imageDataVTK.GetDimensions()
  #spacing = imageDataVTK.GetSpacing()
  #origin  = imageDataVTK.GetOrigin()
  #print spacing, origin, dimensions
  ##fem.SetImagingDimensions( dimensions ,origin,spacing) 

  #image_point_data = imageDataVTK.GetPointData() 
  ## write numpy to disk in matlab
  ##  indexing is painful.... reshape to dimensions and transpose 2d dimensions only
  #scipyio.savemat( output_filename, {'spacing':spacing, 'origin':origin,'image':image_data.reshape(dimensions,order='F').transpose(1,0,2)})

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--file_name",
                  action="store", dest="file_name", default=None,
                  help="converting/this/file to converting/this/file.vtk", metavar = "FILE")
(options, args) = parser.parse_args()
if (options.file_name):
  OutputFileName = options.file_name.split('.').pop(0) + '.vtk'
  ConvertGadgetronVTK(options.file_name,OutputFileName )
else:
  parser.print_help()
  print options
