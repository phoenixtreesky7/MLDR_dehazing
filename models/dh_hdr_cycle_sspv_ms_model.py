import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import dh_hdr_networks
from epdn import epdn_networks
import math
from ECLoss.ECLoss import DCLoss
from TVLoss.TVLossL1 import TVLossL1 
from models.gradient import gradient
class DH_HDR_CYC_SSPV(BaseModel):
    def name(self):
        return 'DH_HDR_CYC_SSPV'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.gpu_ids
        self.hl = opt.hl
        self.opt = opt
        self.dh_real = opt.dh_real

        # specify the training losses you want to print out. The program will call base_model.get_current_losses  loss_style_B + self.loss_content_A + self.loss_content_B
        self.loss_names = ['D_A1', 'G_A1', 'cycle_A1', 'idt_A1', 'D_B2', 'G_B2', 'cycle_B2', 'idt_B2', 'G_VGG_recA1', 'G_VGG_recB2', 'G_VGG_fake_B1', 'G_VGG_fake_A2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A1', 'real_B1', 'fake_B1', 'rec_A1']
        visual_names_B = ['real_A2', 'real_B2', 'fake_A2', 'rec_B2']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A1')
            visual_names_B.append('idt_B2')

        self.visual_names = visual_names_A + visual_names_B
        if self.dh_real:
            self.loss_names += ['G_C', 'gradient_deh_C','content_deh_C']
            visual_names_C = ['real_C', 'deh_C']
            self.visual_names += visual_names_C
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_E_A', 'G_E_B', 'G_D_A', 'G_D_B', 'D_A', 'D_B']
            if self.dh_real:
                self.model_names += ['D_C']
        else:  # during test time, only load Gs
            self.model_names = ['G_E_A', 'G_E_B', 'G_D_A', 'G_D_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_E_A = dh_hdr_networks.define_G_E_A(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.n_blocks, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_D_A = dh_hdr_networks.define_G_D_A(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.n_blocks, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_E_B = dh_hdr_networks.define_G_E_B(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.n_blocks, opt.hl, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_D_B = dh_hdr_networks.define_G_D_B(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.n_blocks, opt.hl, opt.which_model_netG, opt.fuse_model, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = dh_hdr_networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = dh_hdr_networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_C = dh_hdr_networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_A2_pool = ImagePool(opt.pool_size)
            self.fake_B1_pool = ImagePool(opt.pool_size)
            self.deh_C_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = dh_hdr_networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionVGG = epdn_networks.VGGLoss(self.gpu_ids)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_E_A.parameters(), self.netG_E_B.parameters(), self.netG_D_A.parameters(), self.netG_D_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_C.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        #AtoB = self.opt.which_direction == 'AtoB'
        self.real_A1 = input['A1'].to(self.device)
        self.real_B1 = input['B1'].to(self.device)
        self.real_A2 = input['A2'].to(self.device)
        self.real_B2 = input['B2'].to(self.device)
        self.image_paths = input['A1_paths']
        if self.dh_real:
            self.real_C = input['C'].to(self.device)



    def forward(self):
        # Forward Cycle #
        # disentanglement #
        self.content_A1 = self.netG_E_A(self.real_A1)
        self.content_B1, self.style_B1 = self.netG_E_B(self.real_B1)
        #self.content_A2 = self.netG_E_A(self.real_A2)
        self.content_B2, self.style_B2 = self.netG_E_B(self.real_B2)
        if self.dh_real:
            self.content_C, self.style_C = self.netG_E_B(self.real_C)  ## 用于特征聚合

        # translation #
        self.fake_B1 = self.netG_D_B(self.real_A1, self.content_A1, self.style_B2)
        self.fake_A2 = self.netG_D_A(self.real_B2, self.content_B2)
        if self.dh_real:
            self.deh_C = self.netG_D_A(self.real_C, self.content_C)
            self.content_deh_C = self.netG_E_A(self.deh_C)

        # identity #
        self.idt_A1 = self.netG_D_A(self.real_A1, self.content_A1)
        self.idt_B2 = self.netG_D_B(self.real_B2, self.content_B2, self.style_B2)

        # Backward Cycle #
        # disentanglement #
        self.content_fake_A2 = self.netG_E_A(self.fake_A2)
        self.content_fake_B1, self.style_fake_B1 = self.netG_E_B(self.fake_B1)

        # translation #
        self.rec_A1 = self.netG_D_A(self.real_A1, self.content_fake_B1)
        self.rec_B2 = self.netG_D_B(self.real_B2, self.content_fake_A2, self.style_fake_B1)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B1 = self.fake_B1_pool.query(self.fake_B1)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B2, fake_B1)

    def backward_D_B(self):
        fake_A2 = self.fake_A2_pool.query(self.fake_A2)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A1, fake_A2)

    def backwork_D_C(self):
        #deh_C = self.deh_C_pool.query(self.deh_C)
        #self.loss_D_C = self.backward_D_basic(self.netD_C, self.real_A, deh_C)
        deh_C = self.deh_C_pool.query(self.deh_C)
        dehaze_real_A1_real_C_cat = torch.cat((self.real_A1, self.real_C), dim=1)
        dehaze_real_A1_deh_C_cat = torch.cat((self.real_A1, deh_C), dim=1)
        self.loss_D_C = self.backward_D_basic(self.netD_C, dehaze_real_A1_real_C_cat, dehaze_real_A1_deh_C_cat)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_content
        lambda_S = self.opt.lambda_style
        
        # Identity loss
        if lambda_idt > 0:
            # G_D_A should be identity if real_A is fed.
            self.loss_idt_A1 = self.criterionIdt(self.idt_A1, self.real_A1) * lambda_B * lambda_idt
            # G_D_B should be identity if real_B is fed.
            self.loss_idt_B2 = self.criterionIdt(self.idt_B2, self.real_B2) * lambda_A * lambda_idt
            # G_D_A should be identity if real_A is fed.
            self.loss_fake_B1 = 0#self.criterionIdt(self.fake_B1, self.real_B1) * lambda_B * lambda_idt
            # G_D_B should be identity if real_B is fed.
            self.loss_fake_A2 = self.criterionIdt(self.fake_A2, self.real_A2) * lambda_A * lambda_idt
        else:
            self.loss_idt_A1 = 0
            self.loss_idt_B2 = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A1 = self.criterionGAN(self.netD_A(self.fake_B1), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B2 = self.criterionGAN(self.netD_B(self.fake_A2), True)
        
        # Forward cycle loss
        self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A1) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B2 = self.criterionCycle(self.rec_B2, self.real_B2) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A1 + self.loss_G_B2 + self.loss_cycle_A1 + self.loss_cycle_B2 + self.loss_idt_A1 + self.loss_idt_B2


        # Vgg loss
        self.loss_G_VGG_recA1 = self.criterionVGG(self.rec_A1, self.real_A1) * self.opt.lambda_vgg
        self.loss_G_VGG_recB2 = self.criterionVGG(self.rec_B2, self.real_B2) * self.opt.lambda_vgg
        self.loss_G_VGG_fake_B1 = 0 # self.criterionVGG(self.fake_B1, self.real_A1) * self.opt.lambda_vgg
        self.loss_G_VGG_fake_A2 = self.criterionVGG(self.fake_A2, self.real_B2) * self.opt.lambda_vgg
        self.loss_G += self.loss_G_VGG_recA1 + self.loss_G_VGG_recB2 + self.loss_G_VGG_fake_B1 + self.loss_G_VGG_fake_A2

        # Gradient loss
        self.gradient_real_A1 = gradient(self.real_A1)
        self.gradient_real_B2 = gradient(self.real_B2)
        self.gradient_fake_B1 = gradient(self.fake_B1)
        self.gradient_fake_A2 = gradient(self.fake_A2)
        self.gradient_rec_A1 = gradient(self.rec_A1)
        self.gradient_rec_B2 = gradient(self.rec_B2)

        self.loss_gradient_fake_B1 = 0.2 * self.criterionIdt(self.gradient_real_A1, self.gradient_fake_B1)
        self.loss_gradient_fake_A2 = 0.2 * self.criterionIdt(self.gradient_real_B2, self.gradient_fake_A2)
        self.loss_gradient_rec_A1 = 0.2 * self.criterionIdt(self.gradient_real_A1, self.gradient_rec_A1)
        self.loss_gradient_rec_B2 = 0.2 * self.criterionIdt(self.gradient_real_B2, self.gradient_rec_B2)

        self.loss_G += self.loss_gradient_fake_B1 + self.loss_gradient_fake_A2 + self.loss_gradient_rec_A1 + self.loss_gradient_rec_B2

        # Feature Identity loss


        if self.dh_real:
            # GAN loss D_B(G_B(C))
            self.loss_G_C = self.criterionGAN(self.netD_C(torch.cat((self.real_A1, self.deh_C), dim=1)), True)
            
            self.gradient_real_C = gradient(self.real_C)
            self.gradient_deh_C = gradient(self.deh_C)
            self.loss_gradient_deh_C = 0.5 * self.criterionIdt(self.gradient_real_C, self.gradient_deh_C)

            self.loss_content_deh_C = self.criterionIdt(self.content_C[3], self.content_deh_C[3]) * lambda_B * lambda_idt

            self.loss_G += self.loss_G_C + self.loss_gradient_deh_C + self.loss_content_deh_C

        #if self.dh_real:
            #self.loss_DCP_deh_C =  DCLoss((self.deh_C+1)/2, self.opt)
            #self.loss_TV_deh_C = lambda_T*TVLossL1(self.deh_C)
            #self.loss_G += self.loss_DCP_deh_C + self.loss_DCP_deh_C

        #n_ds = 3
        #esum = 0
        #for i in range(self.hl):
        #    esum += math.exp(n_ds-i)

        #self.loss_style_B = 0
        #self.loss_content_A = 0
        #self.loss_content_B = 0
        #for i in range(self.hl):
        #    #print('self.hl-i', n_ds-i)
        #    self.loss_style_B += self.criterionMSE(self.style_B[n_ds-i], self.style_fake_B[n_ds-i]) * lambda_S * (math.exp(n_ds-i)/esum)
        #    #self.loss_content_A += self.criterionIdt(self.content_A[n_ds-i], self.content_fake_B[n_ds-i]) * lambda_C
            #self.loss_content_B += self.criterionIdt(self.content_B[n_ds-i], self.content_fake_A[n_ds-i]) * lambda_C

        #self.loss_G += self.loss_style_B #+ self.loss_content_A + self.loss_content_B
        self.loss_G.backward()

    def optimize_parameters(self, opt):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C], False)

        self.set_requires_grad([self.netG_E_A, self.netG_E_B, self.netG_D_A, self.netG_D_B], True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
