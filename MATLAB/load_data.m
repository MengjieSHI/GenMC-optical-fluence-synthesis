[filename, filepath]=uigetfile('*.raw','Please choose file',RawDataPath);
obj =  strcat(filepath,filename);

frameNumber_char = filename(end-7:end-4); % find frameNumber last 3-4 char
if contains( frameNumber_char,'_')
    frameNumber_char = strip(frameNumber_char, '_');
else 
    frameNumber_char = frameNumber_char;
end
frameName_char = filename(end-20:end-15);
frameNumberUS = str2num(frameNumber_char);
transducerElements = 128;
tmpDataUS = readAcousticXData(obj, transducerElements, frameNumberUS);