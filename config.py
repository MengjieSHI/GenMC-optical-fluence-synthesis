import argparse

def get_options(parser=argparse.ArugmentParser()):
    
    parser.add_argument("--dataset", type=str, default="256_data", help="number of datasets")
    parser.add_argument("--model",type=str, default="pix2pix", help="name of models")
    parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=int, default=0.0002, help="learning rate")
    parser.add_argument("--lr_lambda", type=int, default=1, help="scheduler decay rate")
    parser.add_argument("--bilinear", action="store_true", default=True, help='Use bilinear upsampling')
    parser.add_argument("--output", action="store_true", default=True, help="shows output")
    args = parser.parse_args()

    if args.output: 
        print(f'dataset:{args.dataset}')
        print(f'num_epochs:{args.epochs}')
        print(f'num_batch_size:{args.batch_size}')
        print(f'learning_rate:{args.lr}')
        print(f'scheduler_decay_rate:{args.lr_lambda}')
        print(f'bilinear:{args.bilinear}')
        
    return args

if __name__=='__main__':
    opt = get_options()