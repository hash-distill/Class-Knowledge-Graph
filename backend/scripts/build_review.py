import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review.builder import build_lesson

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Build a cited lesson review from completed ASR and video-slide observations')
    parser.add_argument('--lesson-id',default='lesson1-real')
    parser.add_argument('--video',type=Path,default=Path(__file__).resolve().parents[2]/'speech-models/lesson1.mp4')
    parser.add_argument('--media-dir',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts/lesson1-real')
    parser.add_argument('--title',default='正弦定理 · 课堂复盘')
    parser.add_argument('--allow-no-speech',action='store_true',help='Explicitly publish a visual-evidence lesson with unrecognized speech; never synthesize a transcript')
    args=parser.parse_args()
    bundle=build_lesson(args.lesson_id,args.video,args.media_dir,args.title,allow_no_speech=args.allow_no_speech)
    print(bundle['report']['summary'])
    print('Saved:',args.media_dir/'bundle.json')
