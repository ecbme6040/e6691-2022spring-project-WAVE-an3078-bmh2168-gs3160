from utils import *
import wave

def split_all(path,target):
    """
   Splits all audio from inside path recursively into chunks of 1.1 seconds. Puts them into the target folder
    
    """
    total=0
    all_paths=get_all_paths(path,'wav')
    for file_index,p in enumerate(all_paths):
        with wave.open(p, "rb") as infile:
            # get file data
            nchannels = infile.getnchannels()
            sampwidth = infile.getsampwidth()
            framerate = infile.getframerate()
            frames = infile.getnframes()
            rate = infile.getframerate()
            duration = frames / float(rate)
            print('duration',duration,'s')
            for i in range(int(duration/1.1)-1):
                print('file',file_index,i)
                infile.setpos(int( framerate * i * 1.1))
                # extract data
                data = infile.readframes(int(1.1 * framerate))

                # write the extracted data to a new file
                with wave.open(target+'/'+str(file_index)+'_'+str(i)+'.wav', 'w') as outfile:
                    outfile.setnchannels(nchannels)
                    outfile.setsampwidth(sampwidth)
                    outfile.setframerate(framerate)
                    outfile.setnframes(int(len(data) / sampwidth))
                    outfile.writeframes(data)
                    total+=1
    print('generated',total,'files')