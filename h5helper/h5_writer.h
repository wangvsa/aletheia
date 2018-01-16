 
#ifndef H5_WRITER_H
#define H5_WRITER_H

#include "hdf5.h"

#define DATASETNAME "DATA" 
#define RANK  2


herr_t write_h5_data(char *filename, float *data, int dim0, int dim1) {
    hsize_t     dims[2];   
 
    hid_t       file_id, dataset_id;        /* handles */
    hid_t       dataspace_id;

    herr_t      status;                             
   
    
    file_id = H5Fcreate (filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    dims[0] = dim0;
    dims[1] = dim1;

    dataspace_id = H5Screate_simple(RANK, dims, NULL); 

    dataset_id = H5Dcreate2 (file_id, DATASETNAME, H5T_NATIVE_FLOAT, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite (dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    status = H5Sclose (dataspace_id);
    status = H5Dclose (dataset_id);
    status = H5Fclose (file_id);
    return status;
}


#endif
