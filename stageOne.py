import torch
import torch.nn as nn
import argparse
from utils.tools import *
import json
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from model import VTF
import os
from dataloader.dataset_stageone import trailer_multimodal_features
import pickle as pkl
def main(args):
    traindata = trailer_multimodal_features(phase='train',
                                            data_dir=args.data_dir,
                                            feature_dir=args.feature_dir,
                                            visual_feature_dir=args.visual_feature_dir,
                                            visual_feature_file=args.visual_feature_version+"_train.pkl",
                                            cau_visual = args.cau_visual,
                                            audio_feature_file=args.audio_feature_file,
                                            cau_audio=args.cau_audio,
                                            text_feature_dir = args.text_feature_dir,
                                            text_token_file=args.text_token_file,
                                            text_feature_file=args.text_feature_file,
                                            cau_text=args.cau_text,
                                            num_classes=args.num_categories,
                                            )

    valdata = trailer_multimodal_features(phase='val',
                                          data_dir=args.data_dir,
                                          feature_dir=args.feature_dir,
                                          visual_feature_dir=args.visual_feature_dir,
                                          visual_feature_file=args.visual_feature_version + "_val.pkl",
                                          cau_visual=args.cau_visual,
                                          audio_feature_file=args.audio_feature_file,
                                          cau_audio=args.cau_audio,
                                          text_feature_dir=args.text_feature_dir,
                                          text_token_file=args.text_token_file,
                                          text_feature_file=args.text_feature_file,
                                          cau_text=args.cau_text,
                                          num_classes=args.num_categories,
                                          )

    testdata = trailer_multimodal_features(phase='test',
                                           data_dir=args.data_dir,
                                           feature_dir=args.feature_dir,
                                           visual_feature_dir=args.visual_feature_dir,
                                           visual_feature_file=args.visual_feature_version + "_test.pkl",
                                           cau_visual=args.cau_visual,
                                           audio_feature_file=args.audio_feature_file,
                                           cau_audio=args.cau_audio,
                                           text_feature_dir=args.text_feature_dir,
                                           text_token_file=args.text_token_file,
                                           text_feature_file=args.text_feature_file,
                                           cau_text=args.cau_text,
                                           num_classes=args.num_categories,
                                           )
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=args.batch_size, shuffle=False)

    #model = VTF.VectorTransformer(args.shot_num*args.input_dim, args.output_dim)
    model = VTF.VectorTransformer( args.input_dim, args.output_dim)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    ori_w_t_dic = {}
    for epoch in range(args.num_epoch):
        total_loss = 0.0
        model.train()
        feat_dict = {}
        aug_feat_dict = {}

        ori_mean_v_dic = {}
        for batch_idx, sample in enumerate(train_loader):
            movie_id, label = sample['movie_id'], sample['label_onehot']
            visual_feature_sam = sample['visual_feature'].to(device).type(FloatTensor)
            aug_visual_feature_sam = sample['aug_visual_feature'].to(device).type(FloatTensor)
            topic_feature_dic = sample['topic_feature']
            topic_feature = topic_feature_dic['topic'].to(device).type(FloatTensor)
            tpoic_w = topic_feature_dic['w'].to(device).type(FloatTensor)

            W_expanded = tpoic_w.unsqueeze(2).expand(-1, -1, 768)
            W_B = torch.mul(W_expanded, topic_feature)
            W_t = torch.sum(W_B, dim=1)

            #visual_feature = visual_feature.view(len(visual_feature),-1)
            visual_feature_in = torch.mean(visual_feature_sam, dim=1)
            #visual_feature_in = visual_feature_sam.view(len(visual_feature_sam), -1)
            #aug_visual_feature = aug_visual_feature.view(len(visual_feature),-1)
            aug_visual_feature_in = torch.mean(aug_visual_feature_sam, dim=1)
            #aug_visual_feature_in = aug_visual_feature_sam.view(len(aug_visual_feature_sam), -1)

            visual_feature = model(visual_feature_in)
            aug_visual_feature = model(aug_visual_feature_in)

            loss_fac_M1 = factorization_loss(visual_feature, aug_visual_feature, args.factorization_lambda)
            loss_fac_M2_ori = factorization_loss(visual_feature, visual_feature, (1-args.factorization_lambda))
            loss_fac_M2_aug = factorization_loss(aug_visual_feature, aug_visual_feature, (1-args.factorization_lambda))
            loss_t_i = topic_sim_loss(visual_feature, topic_feature, tpoic_w)

            loss = 0.5 * loss_fac_M1 + 0.5*(1 * loss_fac_M2_ori + 1 * loss_fac_M2_aug) + 0.8 * loss_t_i

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if epoch + 1 == args.num_epoch:
                save_visual_feature = visual_feature.detach()
                sava_aug_feature = aug_visual_feature.detach()
                #
                save_visual_feature_in = visual_feature_in.detach()
                save_W_t = W_t.detach()

                for i in range(len(movie_id)):
                    feat_dict[movie_id[i]] = save_visual_feature[i].cpu().numpy().reshape(1, -1)
                    aug_feat_dict[movie_id[i]] = sava_aug_feature[i].cpu().numpy().reshape(1, -1)
                    ori_mean_v_dic[movie_id[i]]  = save_visual_feature_in[i].cpu().numpy().reshape(1, -1)
                    ori_w_t_dic[movie_id[i]]  = save_W_t[i].cpu().numpy().reshape(1, -1)
                save_dict_path = os.path.join(args.visual_feature_dir, 'cau_ViT-L-14-336_train.pkl')
                with open(save_dict_path, "wb") as handle:
                    pkl.dump(feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
                save_dict_path = os.path.join(args.visual_feature_dir, 'cau_aug_ViT-L-14-336_train.pkl')
                with open(save_dict_path, "wb") as handle:
                    pkl.dump(aug_feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

                save_dict_path = os.path.join(args.visual_feature_dir, 'mean_ViT-L-14-336_train.pkl')
                with open(save_dict_path, "wb") as handle:
                    pkl.dump(ori_mean_v_dic, handle, protocol=pkl.HIGHEST_PROTOCOL)

        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss}")
        model.eval()

        with torch.no_grad():
            feat_dict = {}
            aug_feat_dict = {}

            ori_mean_v_dic = {}
            total_loss = 0.0
            for sample in val_loader:
                movie_id, label = sample['movie_id'], sample['label_onehot']
                visual_feature_sam = sample['visual_feature'].to(device).type(FloatTensor)
                aug_visual_feature_sam = sample['aug_visual_feature'].to(device).type(FloatTensor)
                topic_feature_dic = sample['topic_feature']
                topic_feature = topic_feature_dic['topic'].to(device).type(FloatTensor)
                tpoic_w = topic_feature_dic['w'].to(device).type(FloatTensor)

                W_expanded = tpoic_w.unsqueeze(2).expand(-1, -1, 768)
                W_B = torch.mul(W_expanded, topic_feature)
                W_t = torch.sum(W_B, dim=1)


                visual_feature_in = torch.mean(visual_feature_sam, dim=1)

                aug_visual_feature_in = torch.mean(aug_visual_feature_sam, dim=1)


                visual_feature = model(visual_feature_in)
                aug_visual_feature = model(aug_visual_feature_in)

                loss_fac_M1 = factorization_loss(visual_feature, aug_visual_feature, args.factorization_lambda)
                loss_fac_M2_ori = factorization_loss(visual_feature, visual_feature, (1-args.factorization_lambda))
                loss_fac_M2_aug = factorization_loss(aug_visual_feature, aug_visual_feature, (1-args.factorization_lambda))
                loss_t_i = topic_sim_loss(visual_feature, topic_feature, tpoic_w)
                #loss_t_i = sim_loss(visual_feature, W_t)

                loss = 0.5 * loss_fac_M1 + 0.5*(1 * loss_fac_M2_ori + 1 * loss_fac_M2_aug) + 0.8 * loss_t_i
                #loss = loss_fac_M1 + 0.2 * (loss_fac_M2_ori + loss_fac_M2_aug)
                total_loss += loss.item()
                if epoch + 1 == args.num_epoch:
                    save_visual_feature = visual_feature.detach()
                    sava_aug_feature = aug_visual_feature.detach()
                    #
                    save_visual_feature_in = visual_feature_in.detach()
                    save_W_t = W_t.detach()
                    for i in range(len(movie_id)):
                        feat_dict[movie_id[i]] = save_visual_feature[i].cpu().numpy().reshape(1, -1)
                        aug_feat_dict[movie_id[i]] = sava_aug_feature[i].cpu().numpy().reshape(1, -1)
                        #
                        #print(i)
                        ori_mean_v_dic[movie_id[i]] = save_visual_feature_in[i].cpu().numpy().reshape(1, -1)
                        ori_w_t_dic[movie_id[i]] = save_W_t[i].cpu().numpy().reshape(1, -1)
                    save_dict_path = os.path.join(args.visual_feature_dir, 'cau_ViT-L-14-336_val.pkl')
                    with open(save_dict_path, "wb") as handle:
                        pkl.dump(feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
                    save_dict_path = os.path.join(args.visual_feature_dir, 'cau_aug_ViT-L-14-336_val.pkl')
                    with open(save_dict_path, "wb") as handle:
                        pkl.dump(aug_feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

                    save_dict_path = os.path.join(args.visual_feature_dir, 'mean_ViT-L-14-336_val.pkl')
                    with open(save_dict_path, "wb") as handle:
                        pkl.dump(ori_mean_v_dic, handle, protocol=pkl.HIGHEST_PROTOCOL)
            val_loss = total_loss / len(val_loader)
            print(f"Epoch {epoch + 1}: Validation Loss: {val_loss}")
            total_loss = 0.0
            feat_dict = {}
            aug_feat_dict = {}

            ori_mean_v_dic = {}
            for sample in test_loader:
                movie_id, label = sample['movie_id'], sample['label_onehot']
                visual_feature_sam = sample['visual_feature'].to(device).type(FloatTensor)
                aug_visual_feature_sam = sample['aug_visual_feature'].to(device).type(FloatTensor)
                topic_feature_dic = sample['topic_feature']
                topic_feature = topic_feature_dic['topic'].to(device).type(FloatTensor)
                tpoic_w = topic_feature_dic['w'].to(device).type(FloatTensor)

                W_expanded = tpoic_w.unsqueeze(2).expand(-1, -1, 768)
                W_B = torch.mul(W_expanded, topic_feature)
                W_t = torch.sum(W_B, dim=1)

                # visual_feature = visual_feature.view(len(visual_feature),-1)
                visual_feature_in = torch.mean(visual_feature_sam, dim=1)
                #visual_feature_in = visual_feature_sam.view(len(visual_feature_sam), -1)
                # aug_visual_feature = aug_visual_feature.view(len(visual_feature),-1)
                aug_visual_feature_in = torch.mean(aug_visual_feature_sam, dim=1)
                #aug_visual_feature_in = aug_visual_feature_sam.view(len(aug_visual_feature_sam), -1)

                visual_feature = model(visual_feature_in)
                aug_visual_feature = model(aug_visual_feature_in)

                # loss_fac_M1 = factorization_loss(visual_feature, aug_visual_feature, 0.6)
                # loss_fac_M2_ori = factorization_loss(visual_feature, visual_feature, 0.4)
                # loss_fac_M2_aug = factorization_loss(aug_visual_feature, aug_visual_feature, 0.4)
                loss_fac_M1 = factorization_loss(visual_feature, aug_visual_feature, args.factorization_lambda)
                loss_fac_M2_ori = factorization_loss(visual_feature, visual_feature, (1-args.factorization_lambda))
                loss_fac_M2_aug = factorization_loss(aug_visual_feature, aug_visual_feature, (1-args.factorization_lambda))
                loss_t_i = topic_sim_loss(visual_feature, topic_feature, tpoic_w)
                #loss_t_i = sim_loss(visual_feature, W_t)

                loss = 0.5 * loss_fac_M1 + 0.5*(1 * loss_fac_M2_ori + 1 * loss_fac_M2_aug) + 0.8 * loss_t_i
                total_loss += loss.item()
                if epoch + 1 == args.num_epoch:
                    save_visual_feature = visual_feature.detach()
                    sava_aug_feature = aug_visual_feature.detach()
                    #
                    save_visual_feature_in = visual_feature_in.detach()
                    save_W_t = W_t.detach()
                    for i in range(len(movie_id)):
                        feat_dict[movie_id[i]] = save_visual_feature[i].cpu().numpy().reshape(1, -1)
                        aug_feat_dict[movie_id[i]] = sava_aug_feature[i].cpu().numpy().reshape(1, -1)
                        #
                        ori_mean_v_dic[movie_id[i]] = save_visual_feature_in[i].cpu().numpy().reshape(1, -1)
                        ori_w_t_dic[movie_id[i]]  = save_W_t[i].cpu().numpy().reshape(1, -1)
                    save_dict_path = os.path.join(args.visual_feature_dir, 'cau_ViT-L-14-336_test.pkl')
                    with open(save_dict_path, "wb") as handle:
                        pkl.dump(feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
                    save_dict_path = os.path.join(args.visual_feature_dir, 'cau_aug_ViT-L-14-336_test.pkl')
                    with open(save_dict_path, "wb") as handle:
                        pkl.dump(aug_feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

                    save_dict_path = os.path.join(args.visual_feature_dir, 'mean_ViT-L-14-336_test.pkl')
                    with open(save_dict_path, "wb") as handle:
                        pkl.dump(ori_mean_v_dic, handle, protocol=pkl.HIGHEST_PROTOCOL)
            val_loss = total_loss / len(val_loader)
            print(f"Epoch {epoch + 1}: Test Loss: {val_loss}")

    save_W_t_path = os.path.join('./data/Douban2595/features/Topic_features', 'w_topic_BERTopic.pkl')
    with open(save_W_t_path, "wb") as handle:
        pkl.dump(ori_w_t_dic, handle, protocol=pkl.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #DATASET
    DATASET_NAME = 'Douban2595'
    DATA_DIR = 'G:/data/DouBan2595/shot_frame'
    FEATURE_DIR = os.path.join('data', DATASET_NAME, 'features')
    VISUAL_FOLDER = os.path.join(FEATURE_DIR, 'Visual_features_L')
    VISUAL_FEATURE_VERSION = 'ViT-L-14-336'
    # AUDIO_FOLDER = os.path.join(FEATURE_DIR, 'Audio_features')
    # AUDIO_FEATURE_FILE = os.path.join(FEATURE_DIR, 'Audio_features/ori_audio_PANNs.pkl')
    AUDIO_FOLDER = None
    AUDIO_FEATURE_FILE = None
    TEXT_FOLDER = os.path.join(FEATURE_DIR, 'Topic_features')
    TEXT_FEATURE_FILE = os.path.join(FEATURE_DIR, 'Topic_features/topic_BERTopic.pkl')
    #META_FOLDER = os.path.join(FEATURE_DIR, 'Meta_features')
    META_FOLDER = None

    num_category = {'MovieNet1100':20,'Douban2595':26}
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='path to save feature files')
    parser.add_argument('--feature_dir', type=str, default=FEATURE_DIR, help='path to save feature files')
    parser.add_argument('--visual_feature_dir', type=str, default=VISUAL_FOLDER, help='path to visual features')
    parser.add_argument('--visual_feature_version', type=str, default="ViT-L-14-336")
    parser.add_argument('--cau_visual', type=str2bool, default=False)
    parser.add_argument('--text_feature_dir', type=str, default=None)
    parser.add_argument('--text_feature_file', type=str, default=TEXT_FEATURE_FILE)
    parser.add_argument('--cau_text', type=str2bool, default=False)
    parser.add_argument('--audio_feature_dir', type=str, default=None)
    parser.add_argument('--audio_feature_file', type=str,default=None)
    parser.add_argument('--cau_audio', type=str2bool, default=False)
    parser.add_argument('--text_token_file',type=str,default=None)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--output_dim', type=int, default=768)
    parser.add_argument('--shot_num', type=int, default=8)
    parser.add_argument('--num_categories', type=int, default=num_category[DATASET_NAME], help='num_categories')
    parser.add_argument('--num_epoch', type=int, default=17, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--factorization_lambda', type=float, default=0.8, help='lambda of factorization_loss')
    args = parser.parse_args()
    print(args)
    main(args)
    #+++++++++++++++++++++++++++++++
