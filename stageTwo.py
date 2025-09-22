import os
import argparse

import torch

from utils.tools import *
import json
from dataloader.dataset_stagetwo import trailer_multimodal_features
import torch.optim as optim
from model import JEM
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.metrics import coverage_error, label_ranking_loss
from utils.tools import sigmoid, str2bool
import math


def train(epoch, args):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        #print(batch_idx)
        if batch_idx == 72:
            print('1111')
            #input()
        movie_id, label = sample['movie_id'],sample['label_onehot']
        f_one= {}
        f_one['visual_feature'] = sample['visual_feature'].to(device).type(FloatTensor)
        # f_one['text_feature'] = sample['text_feature'].to(device).type(FloatTensor)
        # f_one['audio_feature'] = sample['audio_feature'].to(device).type(FloatTensor)
        f_one['cau_visual_feature'] = sample['cau_visual_feature'].to(device).type(FloatTensor)
        f_one['VTM_visual_feature'] = sample['VTM_visual_feature'].to(device).type(FloatTensor)
        f_one['topic_feature'] = sample['topic_feature'].to(device).type(FloatTensor)
        f_one['VTM_topic_feature'] = sample['VTM_topic_feature'].to(device).type(FloatTensor)

        # f_one['cau_text_feature'] = sample['cau_text_feature'].to(device).type(FloatTensor)
        # f_one['cau_audio_feature'] = sample['cau_audio_feature'].to(device).type(FloatTensor)
        # f_one['meta_glo'] = sample['meta_glo'].to(device).type(FloatTensor)
        # f_one['meta_loc'] = sample['meta_loc'].to(device).type(FloatTensor)

        label = label.type(LongTensor)
        type_index = torch.cat((torch.zeros(f_one['visual_feature'].shape[-1], dtype=torch.int64), torch.ones(f_one['cau_visual_feature'].shape[-1], dtype=torch.int64)), dim=0).to(device)
        pe = generate_positional_encoding(args.hidden_dim, args.ori_len + args.cau_len ).to(device)
        optimizer.zero_grad()
        output = model(f_one, type_index, pe, args.lambda_f, args.lambda_m)
        loss = F.binary_cross_entropy_with_logits(output, label.float())
        loss.backward()
        optimizer.step()

        if batch_idx % args.process_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(label), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
def val(args):
    global val_loader

    return_predictions = []
    return_labels = []

    with torch.no_grad():
        model.eval()
        val_loss = 0
        coverageloss = 0
        rl = 0
        for batch_idx, sample in enumerate(val_loader):
            movie_id, label = sample['movie_id'], sample['label_onehot']
            f_one = {}
            f_one['visual_feature'] = sample['visual_feature'].to(device).type(FloatTensor)
            f_one['cau_visual_feature'] = sample['cau_visual_feature'].to(device).type(FloatTensor)
            f_one['VTM_visual_feature'] = sample['VTM_visual_feature'].to(device).type(FloatTensor)
            f_one['topic_feature'] = sample['topic_feature'].to(device).type(FloatTensor)
            f_one['VTM_topic_feature'] = sample['VTM_topic_feature'].to(device).type(FloatTensor)

            label = label.type(LongTensor)
            type_index = torch.cat((torch.zeros(f_one['visual_feature'].shape[-1], dtype=torch.int64), torch.ones(f_one['VTM_visual_feature'].shape[-1], dtype=torch.int64)), dim=0).to(device)
            pe = generate_positional_encoding(args.hidden_dim, args.ori_len + args.cau_len).to(device)
            output = model(f_one, type_index, pe, args.lambda_f, args.lambda_m)
            predicted = output.data
            predicted_np = predicted.cpu().numpy()
            label_np = label.cpu().numpy()
            return_predictions.append(predicted_np)
            return_labels.append(label_np)

        predictions_np, labels_np = np.concatenate(return_predictions), np.concatenate(return_labels)

        mAP = average_precision_score(labels_np, sigmoid(predictions_np))
        val_loss /= len(val_loader)
        coverageloss /= len(val_loader)
        rl /= len(val_loader)
        print('\nValidation set: Average loss: {:.4f}, mAP: {:.4f} \n'
              .format(val_loss, mAP + coverageloss + rl))

    return labels_np, output.data, val_loss, mAP
def main(args):
    global BCE_criterion, train_loader, val_loader, model, optimizer
    log_name = os.path.join("logs", args.model_name)
    os.makedirs(log_name, exist_ok=True)

    traindata = trailer_multimodal_features(phase='train',
                                            data_dir=args.data_dir,
                                            feature_dir=args.feature_dir,
                                            visual_feature_dir=args.visual_feature_dir,
                                            visual_feature_file=args.visual_feature_version+"_train.pkl",
                                            cau_visual=args.cau_visual,
                                            audio_feature_dir = args.audio_feature_dir,
                                            audio_feature_file=args.audio_feature_file,
                                            cau_audio= args.cau_audio,
                                            text_feature_dir = args.text_feature_dir,
                                            text_token_file=args.text_token_file,
                                            text_feature_file=args.text_feature_file,
                                            cau_text=args.cau_text,
                                            meta_glo_file=args.meta_glo_file,
                                            meta_loc_file=args.meta_loc_file,
                                            num_classes=args.num_categories,
                                            )
    valdata = trailer_multimodal_features(phase='val',
                                          data_dir=args.data_dir,
                                          feature_dir=args.feature_dir,
                                          visual_feature_dir=args.visual_feature_dir,
                                          visual_feature_file=args.visual_feature_version + "_val.pkl",
                                          cau_visual=args.cau_visual,
                                          audio_feature_dir=args.audio_feature_dir,
                                          audio_feature_file=args.audio_feature_file,
                                          cau_audio=args.cau_audio,
                                          text_feature_dir=args.text_feature_dir,
                                          text_token_file=args.text_token_file,
                                          text_feature_file=args.text_feature_file,
                                          cau_text=args.cau_text,
                                          meta_glo_file=args.meta_glo_file,
                                          meta_loc_file=args.meta_loc_file,
                                          num_classes=args.num_categories,
                                          )



    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, pin_memory=False)

    model = JEM.JEM(args.hidden_dim, args.num_layers, args.num_heads, args.meta_num, args.ori_len, args.a_len, args.cau_len, args.type_num, args.num_categories)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),  lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=0, verbose=True)
    BCE_criterion = nn.BCEWithLogitsLoss()

    best_macro_mAP = 0
    f_logger = open(log_name + "/logger_info.txt", 'w')
    for epoch_org in range(args.num_epoch):
        epoch = epoch_org + 1
        train(epoch, args)
        _, _, val_loss, macro_mAP = val(args)
        #scheduler.step(macro_mAP)

        f_logger.write("epoch-{}: val: {:.4f}; mAP: {:.4f} \n".format(epoch, val_loss, macro_mAP))
        if macro_mAP > best_macro_mAP:
            best_macro_mAP = macro_mAP
            torch.save(model, log_name + "/epoch-best.pkl")
            best_epoch = epoch
        if epoch % args.save_interval == 0:
            print('saving the %d epoch' % (epoch))
            torch.save(model, log_name + "/epoch-%d.pkl" % (epoch))

    # f_logger.write("best epoch num: %d" % best_epoch)
    # f_logger.close()
    #
    # results = vars(args)
    # results.update({'best_epoch_mAP': best_macro_mAP, 'best_epoch': best_epoch})
    #
    # with open(os.path.join(log_name, "train_info.json"), 'w') as f:
    #     json.dump(results, f, indent=2)

    if args.include_test:

        from test import test_main
    class test_ap(object):
        def __init__(self, args):
            self.model_path = log_name
            self.save_results = True
            self.dataloader_name = "trailer_multimodal_features"
            self.ori_len = args.ori_len
            self.cau_len = args.cau_len
            self.hidden_dim = args.hidden_dim
            self.lambda_f = args.lambda_f
            self.lambda_m = args.lambda_m

    test_args = test_ap(args)
    test_main(test_args)




if __name__ == '__main__':
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #DATASET
    DATASET_NAME = 'MovieNet1100'
    #DATA_DIR = 'G:/data/MovieNet1100/shot_frame'
    DATA_DIR = 'G:/data/movieNet/240P/all'
    FEATURE_DIR = os.path.join('data', DATASET_NAME, 'features')
    VISUAL_FOLDER = os.path.join(FEATURE_DIR, 'Visual_features_L')
    VISUAL_FEATURE_VERSION = 'ViT-L-14-336'
    #VISUAL_FEATURE_VERSION = 'TG'
    #VISUAL_FEATURE_VERSION = None
    # AUDIO_FOLDER = os.path.join(FEATURE_DIR, 'Audio_features')
    # AUDIO_FEATURE_FILE = os.path.join(FEATURE_DIR, 'Audio_features/ori_audio_PANNs.pkl')
    AUDIO_FOLDER = None
    AUDIO_FEATURE_FILE = None
    TEXT_FOLDER = os.path.join(FEATURE_DIR, 'Topic_features')
    TEXT_FEATURE_FILE = os.path.join(FEATURE_DIR, 'Topic_features/w_topic_BERTopic.pkl')
    #TEXT_FEATURE_FILE = None
    META_FOLDER = os.path.join(FEATURE_DIR, 'Meta_features')
    # META_GLO_FILE = os.path.join(FEATURE_DIR, 'Meta_features/glo_meta.pkl')
    # META_LOC_FILE = os.path.join(FEATURE_DIR, 'Meta_features/loc_meta.pkl')
    META_GLO_FILE = None
    META_LOC_FILE = None

    num_category = {'MovieNet1100':20,'Douban2595':26}
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='path to save feature files')
    parser.add_argument('--feature_dir', type=str, default=FEATURE_DIR, help='path to save feature files')
    parser.add_argument('--visual_feature_dir', type=str, default=VISUAL_FOLDER, help='path to visual features')
    parser.add_argument('--visual_feature_version', type=str, default=VISUAL_FEATURE_VERSION)
    parser.add_argument('--cau_visual', type=str2bool, default=True)

    parser.add_argument('--text_feature_dir', type=str, default=TEXT_FOLDER)
    parser.add_argument('--text_feature_file',type=str, default=TEXT_FEATURE_FILE)
    parser.add_argument('--cau_text', type=str2bool, default=False)

    parser.add_argument('--audio_feature_dir', type=str, default=AUDIO_FOLDER)
    parser.add_argument('--audio_feature_file', type=str,default=AUDIO_FEATURE_FILE)
    parser.add_argument('--cau_audio', type=str2bool, default=False)

    parser.add_argument('--text_token_file',type=str,default=None)
    parser.add_argument('--meta_glo_file', type=str,default=META_GLO_FILE)
    parser.add_argument('--meta_loc_file', type=str,default=META_LOC_FILE)

    parser.add_argument('--model_name', type=str, default='trailer_visual_audio_text', help='model name')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

    parser.add_argument('--input_dim', type=int, default=2048)
    parser.add_argument('--output_dim', type=int, default=160)


    parser.add_argument('--shot_num', type=int, default=5)

    parser.add_argument('--num_categories', type=int, default=num_category[DATASET_NAME], help='num_categories')
    parser.add_argument('--num_epoch', type=int, default=0, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')

    parser.add_argument('--hidden_dim', type=int, default=1, help='dimensionality of hidden feature')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--num_heads', type=int, default=1, help='num_heads')
    parser.add_argument('--meta_num', type=int, default=13394, help='meta_num')
    parser.add_argument('--ori_len', type=int, default=768, help='v t len')
    parser.add_argument('--a_len', type=int, default=2048, help='max_len')

    parser.add_argument('--cau_len', type=int, default=768, help='cau_len')
    parser.add_argument('--type_num', type=int, default=2, help='type_num')
    parser.add_argument('--lambda_f', type=float, default=0.3, help='lambda_f')
    parser.add_argument('--lambda_m', type=float, default=0.2, help='lambda_m')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--factorization_lambda', type=float, default=0.8, help='lambda of factorization_loss')
    parser.add_argument('--process_interval', type=int, default=10, help='the interval between process print')
    parser.add_argument('--save_interval', type=int, default=5, help='the interval between saved epochs')
    parser.add_argument('--include_test', type=str2bool, default=True, help='do test or not')
    args = parser.parse_args()
    main(args)

