from bot import MugungHwaBot
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mugungwhat Game AI')
    parser.add_argument('--host', default='127.0.0.1', type=str,
                        help='Host for tcp connection')
    
    parser.add_argument('--port', default=65432, type=int,
                        help='Port for tcp connection')

    parser.add_argument('-m', '--motion_threshold', default=1000, type=int,
                        help='Motion Sensitivity threshold')
    parser.add_argument('-f', '--new_face_threshold', default=1.0, type=float,
                        help='New Face threshold')
    parser.add_argument('-p', '--pixel_count_threshold', default=300.0, type=float,
                        help='Number of pixels of motion detection threshold')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show processing images on screen')
    
    args = parser.parse_args()

    m_bot = MugungHwaBot(verbose=args.verbose,
                        motion_threshold=args.motion_threshold,
                        pixel_count_threshold=args.pixel_count_threshold,
                        new_face_threshold=args.new_face_threshold,
                        host=args.host, port=args.port)
    m_bot.start()
