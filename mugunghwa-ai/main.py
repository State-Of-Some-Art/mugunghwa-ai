from bot import MugungHwaBot
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mugungwhat Game AI')
    parser.add_argument('--host', default='127.0.0.1', type=str,
                        help='Host for tcp connection')
    
    parser.add_argument('--port', default=65432, type=int,
                        help='Port for tcp connection')

    parser.add_argument('-t', '--threshold', default=1000, type=int,
                        help='Motion Sensitivity threshold')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show processing images on screen')
    
    args = parser.parse_args()

    m_bot = MugungHwaBot(verbose=args.verbose, motion_threshold=args.threshold, 
                        host=args.host, port=args.port)
    m_bot.start()
