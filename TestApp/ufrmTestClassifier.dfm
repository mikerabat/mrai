object frmClassifierTest: TfrmClassifierTest
  Left = 0
  Top = 0
  Caption = 'Classification Test'
  ClientHeight = 641
  ClientWidth = 822
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -13
  Font.Name = 'Tahoma'
  Font.Style = []
  OldCreateOrder = False
  OnCreate = FormCreate
  OnDestroy = FormDestroy
  DesignSize = (
    822
    641)
  PixelsPerInch = 96
  TextHeight = 16
  object PaintBox1: TPaintBox
    Left = 9
    Top = 172
    Width = 492
    Height = 481
    Anchors = [akLeft, akTop, akRight, akBottom]
    OnPaint = PaintBox1Paint
  end
  object Label1: TLabel
    Left = 127
    Top = 148
    Width = 144
    Height = 16
    Caption = 'Classifiers Learning Error'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'Tahoma'
    Font.Style = []
    ParentFont = False
  end
  object lblLearnError: TLabel
    Left = 284
    Top = 148
    Width = 8
    Height = 16
    Caption = '0'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'Tahoma'
    Font.Style = [fsBold]
    ParentFont = False
  end
  object GroupBox1: TGroupBox
    Left = 8
    Top = 0
    Width = 806
    Height = 142
    Anchors = [akLeft, akTop, akRight]
    Caption = 'Simple Tests'
    TabOrder = 0
    object butCreateGaussSet: TButton
      Left = 14
      Top = 32
      Width = 139
      Height = 25
      Caption = 'Create 2D Gauss set'
      TabOrder = 0
      OnClick = butCreateGaussSetClick
    end
    object butDecissionStump: TButton
      Left = 14
      Top = 72
      Width = 139
      Height = 25
      Caption = 'Decission Stump'
      TabOrder = 1
      OnClick = butDecissionStumpClick
    end
    object butAdaBoost: TButton
      Left = 161
      Top = 72
      Width = 153
      Height = 25
      Caption = 'AdaBoost Decission Stump'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -11
      Font.Name = 'Tahoma'
      Font.Style = []
      ParentFont = False
      TabOrder = 2
      OnClick = butAdaBoostClick
    end
    object butGentleBoost: TButton
      Left = 161
      Top = 32
      Width = 153
      Height = 25
      Caption = 'Gentle boost Decission Stump'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -11
      Font.Name = 'Tahoma'
      Font.Style = []
      ParentFont = False
      TabOrder = 3
      OnClick = butGentleBoostClick
    end
    object butBagging: TButton
      Left = 320
      Top = 32
      Width = 153
      Height = 25
      Caption = 'Bagging Decission Stump'
      TabOrder = 4
      OnClick = butBaggingClick
    end
    object butC45: TButton
      Left = 479
      Top = 32
      Width = 42
      Height = 25
      Caption = 'C4.5'
      TabOrder = 5
      OnClick = butC45Click
    end
    object butNaiveBayes: TButton
      Left = 479
      Top = 72
      Width = 106
      Height = 25
      Caption = 'Naive Bayes'
      TabOrder = 6
      OnClick = butNaiveBayesClick
    end
    object butRBF: TButton
      Left = 320
      Top = 72
      Width = 72
      Height = 25
      Caption = 'RBF'
      TabOrder = 7
      OnClick = butRBFClick
    end
    object butKMean: TButton
      Left = 398
      Top = 72
      Width = 75
      Height = 25
      Caption = 'KMean'
      TabOrder = 8
      OnClick = butKMeanClick
    end
    object butNeuralNet: TButton
      Left = 527
      Top = 32
      Width = 105
      Height = 25
      Caption = 'Neural Network'
      TabOrder = 9
      OnClick = butNeuralNetClick
    end
    object butLDA: TButton
      Left = 686
      Top = 72
      Width = 101
      Height = 25
      Caption = 'Classic LDA'
      TabOrder = 10
      OnClick = butLDAClick
    end
    object chkWeights: TCheckBox
      Left = 15
      Top = 122
      Width = 122
      Height = 17
      Hint = 'Testing the weighted learner routines or the generic weighting'
      Caption = 'Random Weights'
      TabOrder = 11
    end
  end
  object grpFaces: TGroupBox
    Left = 507
    Top = 172
    Width = 307
    Height = 461
    Anchors = [akTop, akRight, akBottom]
    Caption = 'Image Classifiers'
    TabOrder = 1
    object Label2: TLabel
      Left = 24
      Top = 33
      Width = 97
      Height = 16
      Caption = 'Classifier Labels:'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'Tahoma'
      Font.Style = []
      ParentFont = False
    end
    object lblUnseen: TLabel
      Left = 138
      Top = 33
      Width = 4
      Height = 16
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'Tahoma'
      Font.Style = [fsBold]
      ParentFont = False
    end
    object lblOrigLabels: TLabel
      Left = 138
      Top = 55
      Width = 4
      Height = 16
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'Tahoma'
      Font.Style = [fsBold]
      ParentFont = False
    end
    object Label3: TLabel
      Left = 24
      Top = 55
      Width = 89
      Height = 16
      Caption = 'Original Labels:'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'Tahoma'
      Font.Style = []
      ParentFont = False
    end
    object Label4: TLabel
      Left = 3
      Top = 299
      Width = 42
      Height = 16
      Caption = 'FaceDB'
    end
    object butImgRobustFischerLDA: TButton
      Left = 16
      Top = 140
      Width = 281
      Height = 25
      Caption = 'Robust Fischer LDA - Augmented Basis'
      TabOrder = 0
      OnClick = butImgRobustFischerLDAClick
    end
    object chkBlendPart: TCheckBox
      Left = 168
      Top = 82
      Width = 136
      Height = 17
      Caption = 'Blend Part of Image'
      TabOrder = 1
    end
    object rbFisherLDA: TRadioButton
      Left = 24
      Top = 82
      Width = 113
      Height = 17
      Caption = 'Classic Fisher'
      Checked = True
      TabOrder = 2
      TabStop = True
    end
    object rbRobustFischer: TRadioButton
      Left = 24
      Top = 99
      Width = 113
      Height = 17
      Caption = 'Robust'
      TabOrder = 3
    end
    object rbFastRobustFisher: TRadioButton
      Left = 24
      Top = 117
      Width = 113
      Height = 17
      Caption = 'Fast Robust'
      TabOrder = 4
    end
    object Button8: TButton
      Left = 16
      Top = 171
      Width = 281
      Height = 25
      Caption = 'Incremental Fischer LDA'
      TabOrder = 5
      OnClick = butImgRobustFischerLDAClick
    end
    object edFaceDB: TEdit
      Left = 51
      Top = 296
      Width = 246
      Height = 24
      TabOrder = 6
      Text = '.\AdaBoostFaceDBExmpl\'
    end
    object pbBoostProgress: TProgressBar
      Left = 16
      Top = 357
      Width = 257
      Height = 17
      TabOrder = 7
    end
    object btnFaceDb: TButton
      Left = 16
      Top = 326
      Width = 153
      Height = 25
      Caption = 'Adaboost Face Classifier'
      TabOrder = 8
      OnClick = btnFaceBoostingClick
    end
    object chkSaveClassifier: TCheckBox
      Left = 178
      Top = 330
      Width = 114
      Height = 17
      Caption = 'Save Classifier'
      TabOrder = 9
    end
    object butAdaBoostLoad: TButton
      Left = 16
      Top = 380
      Width = 123
      Height = 25
      Caption = 'Load AdaBoost'
      TabOrder = 10
      OnClick = butAdaBoostLoadClick
    end
    object Button6: TButton
      Left = 143
      Top = 109
      Width = 153
      Height = 25
      Caption = 'Fisher LDA'
      TabOrder = 11
      OnClick = Button6Click
    end
    object chkAutoMerge: TCheckBox
      Left = 162
      Top = 380
      Width = 121
      Height = 42
      Caption = 'Auto merge neg.'#13#10'Examples'
      Checked = True
      State = cbChecked
      TabOrder = 12
      WordWrap = True
    end
    object butIntImgTest: TButton
      Left = 216
      Top = 428
      Width = 75
      Height = 25
      Caption = 'Int Img Test'
      TabOrder = 13
      Visible = False
      OnClick = butIntImgTestClick
    end
  end
  object butSVM: TButton
    Left = 646
    Top = 32
    Width = 149
    Height = 25
    Caption = 'Support Vector Machines'
    TabOrder = 2
    OnClick = butSVMClick
  end
  object chkConfidence: TCheckBox
    Left = 8
    Top = 149
    Width = 112
    Height = 17
    Caption = 'Confidence Map'
    TabOrder = 3
    OnClick = chkConfidenceClick
  end
  object sdSaveAdaBoost: TSaveDialog
    DefaultExt = 'cls'
    FileName = 'haarClassifier.cls'
    Title = 'Save AdaBoost Classifier'
    Left = 544
    Top = 536
  end
  object odAdaBoost: TOpenDialog
    DefaultExt = 'cls'
    FileName = 'haarClassifier.cls'
    Title = 'Load Ada Boost Classifier'
    Left = 672
    Top = 544
  end
end
