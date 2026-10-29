function [newData] = readAcousticXData (file, transducerLength, frameNumber)
    fid = fopen(file, 'rb');
    tmpData = fread(fid, 1024*transducerLength*frameNumber, 'int16');
    fclose(fid);
    newData = reshape(tmpData, [1024 transducerLength frameNumber]);
    clear tmpData;
