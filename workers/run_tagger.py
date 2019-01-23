import argparse
import tagger


parser = argparse.ArgumentParser(description="Run tagging processes")
parser.add_argument('--type', help="Either NER, POS, TOKEN")
parser.add_argument('--lang', help="Language, one of ['en', 'de']")
parser.add_argument('--batch', action='store_true', help="Batch process? BOOL")
args = parser.parse_args()


def main(args):
    if args.type == 'NER':
        if args.batch:
            tagger.run_ner_process_batch(args.lang)
        else:
            print("Not implemented: non-batch NER")
    elif args.type == 'POS':
        if args.batch:
            tagger.run_pos_process_batch(args.lang)
        else:
            tagger.run_pos_process(args.lang)
    elif args.type == 'TOKEN':
        if args.batch:
            tagger.run_tokenize_process_batch(args.lang)
        else:
            print("Not implemented: non-batch tokenize")
    else:
        print("Not implemented: %s" % args.type)


if __name__ == '__main__':
    main(args)
