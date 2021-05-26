// ###################################################################
// #### This file is part of the artificial intelligence project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2014, Michael R. . All rights reserved.
// ####
// #### Unless required by applicable law or agreed to in writing, software
// #### distributed under the License is distributed on an "AS IS" BASIS,
// #### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// #### See the License for the specific language governing permissions and
// #### limitations under the License.
// ###################################################################
unit ufrmTestClassifier;

interface

{.$DEFINE INITRANDSEED}  // uncomment if you do not want the same train set

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, BaseClassifier, ExtCtrls, StdCtrls, Matrix, ComCtrls, Haar2DAdaBoost,
  Haar2DImageSweep, Image2DSweep, Types;

type
  TfrmClassifierTest = class(TForm)
    GroupBox1: TGroupBox;
    butCreateGaussSet: TButton;
    PaintBox1: TPaintBox;
    butDecissionStump: TButton;
    butAdaBoost: TButton;
    Label1: TLabel;
    lblLearnError: TLabel;
    butGentleBoost: TButton;
    butBagging: TButton;
    Button6: TButton;
    grpFaces: TGroupBox;
    butImgRobustFisherLDA: TButton;
    Label2: TLabel;
    lblUnseen: TLabel;
    lblOrigLabels: TLabel;
    Label3: TLabel;
    chkBlendPart: TCheckBox;
    rbFisherLDA: TRadioButton;
    rbRobustFisher: TRadioButton;
    rbFastRobustFisher: TRadioButton;
    Button8: TButton;
    butSVM: TButton;
    edFaceDB: TEdit;
    Label4: TLabel;
    pbBoostProgress: TProgressBar;
    btnFaceDb: TButton;
    chkSaveClassifier: TCheckBox;
    sdSaveAdaBoost: TSaveDialog;
    butAdaBoostLoad: TButton;
    odAdaBoost: TOpenDialog;
    butC45: TButton;
    butNaiveBayes: TButton;
    chkAutoMerge: TCheckBox;
    butRBF: TButton;
    butKMean: TButton;
    butNeuralNet: TButton;
    butIntImgTest: TButton;
    butLDA: TButton;
    chkConfidence: TCheckBox;
    chkWeights: TCheckBox;
    procedure butCreateGaussSetClick(Sender: TObject);
    procedure PaintBox1Paint(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure butDecissionStumpClick(Sender: TObject);
    procedure butAdaBoostClick(Sender: TObject);
    procedure butGentleBoostClick(Sender: TObject);
    procedure butBaggingClick(Sender: TObject);
    procedure Button6Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure butImgRobustFisherLDAClick(Sender: TObject);
    procedure butSVMClick(Sender: TObject);
    procedure butIntImgTestClick(Sender: TObject);
    procedure btnFaceBoostingClick(Sender: TObject);
    procedure butAdaBoostLoadClick(Sender: TObject);
    procedure butC45Click(Sender: TObject);
    procedure butNaiveBayesClick(Sender: TObject);
    procedure butRBFClick(Sender: TObject);
    procedure butKMeanClick(Sender: TObject);
    procedure butNeuralNetClick(Sender: TObject);
    procedure butLDAClick(Sender: TObject);
    procedure chkConfidenceClick(Sender: TObject);
  private
    { Private-Deklarationen }
    fFace1, fFace2 : TBitmap;
    fFaceTest : TBitmap;
    fFaceReconstruct : TBitmap;
    fClMapBmp : TBitmap;

    fMinVal, fMaxVal : double;

    fExamples : TCustomLearnerExampleList;
    fClassifier : TCustomClassifier;

    // temporary face classifier
    fSlidingWin : THaar2DSlidingWindow;

    function Weights : TDoubleDynArray;
    procedure MergeNegExamples(const exmplFileName : string; lst : TClassRecList);
    procedure adaBoostImgStep(Sender : TObject; mtx : TDoubleMatrix; actNum, NumImags : integer; const FileName : string);

    procedure boostLearnProgress(Sender : TObject; progress : integer);
    procedure OnPCAReconstruct(Sender : TObject; rec : TDoubleMatrix);
    procedure PaintFaceClassifier;
    procedure TestLearnError;
    procedure TestUnseenImages;
    function Create2DGaussSet(const mean, stddev : Array of Double; const classLabels : Array of integer;
                              numDim : integer; const numExamples : Array of integer) : TCustomLearnerExampleList;
  public
    { Public-Deklarationen }
  end;

var
  frmClassifierTest: TfrmClassifierTest;

implementation

uses BaseMatrixExamples, math, mathutilfunc, SimpleDecisionStump, AdaBoost,
     CustomBooster, Bagging, EnsembleClassifier, FisherBatchLDA, FisherClassifiers,
     ImageDataSet, ImageMatrixConv, jpeg, IncrementalImageDataSet,
     IncrementalFisherLDA, FisherIncrementalClassifiers, BaseIncrementalLearner,
     IntegralImg, Haar2DDataSet, MatrixImageLists, BinaryReaderWriter,
     BaseMathPersistence, DecisionTree45, TreeStructs, NaiveBayes, SVM, RBF, 
     kmeans, NeuralNetwork, JSONReaderWriter, MatrixASMStubSwitch;

{$R *.dfm}

procedure TfrmClassifierTest.adaBoostImgStep(Sender: TObject; mtx: TDoubleMatrix; actNum,
  NumImags: integer; const FileName: string);
var rc : TClassRecList;
    bmp : TBitmap;
    i : Integer;
    ownMtx : IMatrix;
begin
     pbBoostProgress.Position := actNum*100 div NumImags;
     bmp := TMatrixImageConverter.ConvertImage(mtx, ctGrayScale);

     PaintBox1.Canvas.Brush.Color := clBtnFace;
     PaintBox1.Canvas.Brush.Style := bsSolid;
     PaintBox1.Canvas.FillRect(Rect(0, 0, PaintBox1.Width, PaintBox1.Height));
     PaintBox1.Canvas.Draw(0, 0, bmp);

     bmp.Free;

     // so we own the matrix and can use it as an interface
     ownMtx := mtx.Clone;

     rc := fSlidingWin.Classify(ownMtx, 1);
     PaintBox1.Canvas.Brush.Style := bsClear;
     for i := 0 to rc.Count - 1 do
     begin
          with rc.Item[i] do
               PaintBox1.Canvas.Rectangle(x, y, x + Width, y + Height);
     end;

     // add the false positive classifications to the negative example list
     // -> use that for the next learning round so we can reduce the classification error
     if chkAutoMerge.Checked then
        MergeNegExamples(FileName, rc);

     rc.Free;

     for i := 0 to 1000 - 1 do
     begin
          sleep(1);
          Application.ProcessMessages;
     end;

end;

procedure TfrmClassifierTest.boostLearnProgress(Sender: TObject; progress: integer);
begin
     pbBoostProgress.Position := progress;
end;

procedure TfrmClassifierTest.btnFaceBoostingClick(Sender: TObject);
var haarLearner : THaar2DDiscreteAdaBoostLearner;
    learnParams : THaar2DBoostProps;
    ds : THaar2DLearnerExampleListCreator;
    haar2DClassifier : THaar2DAdaBoost;
    counter: Integer;
    cl : integer;
    numCorrectClassified : integer;
begin
     if not DirectoryExists(edFaceDB.Text) then
     begin
          ShowMessage('Face Database directory does not exists');
          exit;
     end;
     
     Randomize;
     learnParams.winWidth := 24;
     learnParams.winHeight := 24;
     learnParams.numColorPlanes := 1;
     learnParams.FeatureTypes := itSumSqr;
     learnParams.baseProps.NumRounds := 100;
     learnParams.baseProps.WeakLearner := TDecisionStumpLearner.Create;
     learnParams.baseProps.PruneToLowestError := False;
     learnParams.baseProps.InitClassSpecificWeights := True;
     learnParams.baseProps.OwnsLearner := True;

     ds := nil;
     haarLearner := THaar2DDiscreteAdaBoostLearner.Create;
     try
        haarLearner.SetProperties(learnParams);

        ds := THaar2DLearnerExampleListCreator.Create(learnParams.winWidth, learnParams.winHeight,
                                                      learnParams.numColorPlanes, learnParams.FeatureTypes);
        ds.InitFromDir(edFaceDB.Text, True);
        haarLearner.Init(ds.LearnerList);
        haarLearner.OnLearnProgress := boostLearnProgress;
        haar2DClassifier := haarLearner.Learn as THaar2DAdaBoost;

        numCorrectClassified := 0;
        for counter := 0 to ds.LearnerList.Count - 1 do
        begin
             cl := haar2DClassifier.Classify(ds.LearnerList[counter]);

             if cl = ds.LearnerList[counter].ClassVal then
                inc(numCorrectClassified);
        end;

        lblLearnError.Caption := Format('%d of %d, %.2f %% ', [numCorrectClassified, ds.LearnerList.Count, numCorrectClassified/ds.LearnerList.Count*100]);
        
     finally
            haarLearner.Free;
            ds.Free;
     end;

     // save the classifier
     if chkSaveClassifier.Checked then
     begin
          if sdSaveAdaBoost.Execute(Handle) then
             //haar2DClassifier.SaveToFile(sdSaveAdaBoost.FileName, TBinaryReaderWriter);
             haar2DClassifier.SaveToFile(sdSaveAdaBoost.FileName, TJsonReaderWriter);
     end;

     try
        // ##################################################
        // #### Load all images and show the classification result
        fSlidingWin := THaar2DSlidingWindow.Create(haar2DClassifier, 2, 2, 0.2, 1, 71/24);
        try
           fSlidingWin.CombineOverlappingReg := True;
           with TIncrementalImageList.Create do
           try
              OnImageStep := adaBoostImgStep;
              Recursive := True;

              ReadListFromDirectoryRaw(edFaceDB.Text, ctGrayScale);
           finally
                  Free;
           end;
        finally
               fSlidingWin.Free;
        end;
     finally
            pbBoostProgress.Position := 0;
            haar2DClassifier.Free;
     end;
end;

procedure TfrmClassifierTest.butNaiveBayesClick(Sender: TObject);
var learner : TNaiveBayesLearner;
    props : TNaiveBayesProps;
begin
     if Assigned(fExamples) then
     begin
          FreeAndNil(fClMapBmp);
          fClassifier.Free;

          props.HistoMin := fMinVal;
          props.HistoMax := fMaxVal;

          props.NumBins := 10;

          learner := TNaiveBayesLearner.Create;
          try
             learner.SetProps(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;


procedure TfrmClassifierTest.butIntImgTestClick(Sender: TObject);
var pic : TPicture;
    bmp : TBitmap;
    img : IMatrix;
    x, y : integer;
begin
     img := TDoubleMatrix.Create(50, 50);

     for y := 0 to 49 do
         for x := 0 to 49 do
             img[x, y] := x;

     with TIntegralImage.Create(img, itSumSqrTilted, 1) do
     try
        ShowMessage(IntToStr(Round(RecSum(10, 10, 10, 10) - RecSum(11, 10, 10, 10))));
        ShowMessage(IntToStr(Round(RecSum(10, 11, 10, 10) - RecSum(9, 11, 10, 10))));


        ShowMessage(IntToStr(Round(RecSum(0, 0, 10, 1))));
        ShowMessage(IntToStr(Round(RecSum(0, 1, 10, 1))));
        ShowMessage(IntToStr(Round(RecSum(0, 2, 10, 1))));

        ShowMessage(IntToStr(Round(RecSum(10, 0, 10, 1))));
        ShowMessage(IntToStr(Round(RecSum(10, 1, 10, 1))));
        ShowMessage(IntToStr(Round(RecSum(10, 2, 10, 1))));

        ShowMessage(IntToStr(Round(RecSum(10, 10, 10, 10))));
        ShowMessage(IntToStr(Round(RecSum(10, 11, 10, 10))));

     finally
            Free;
     end;

     pic := TPicture.Create;
     pic.LoadFromFile('.\Faces\Face1_TestImg.jpg');
     bmp := TBitmap.Create;
     bmp.PixelFormat := pf24bit;
     bmp.SetSize(pic.Width, pic.Height);
     bmp.Canvas.Draw(0, 0, pic.Graphic);
     pic.Free;

     img := TMatrixImageConverter.ConvertImage(bmp, ctGrayScale);
     bmp.Free;
     with TIntegralImage.Create(img, itSumSqrTilted, 1) do
     try
        // check some features
        RecSumSQR(30, 30, 15, 10);

        RecSumSQR(0, 5, 2,2);
        RecsumSQR(-1, 0, 2,2);
        RecSumSQR(160, 160, 50, 50);
        TiltRecSum(20, 30, 30, 20);
     finally
            Free;
     end;

end;

procedure TfrmClassifierTest.butRBFClick(Sender: TObject);
var learner : TRadialBasisLearner;
    props : TRBFProperties;
begin
     if Assigned(fExamples) then
     begin
          fClassifier.Free;
          FreeAndNil(fClMapBmp);

          props.kernel := rbGauss;
          //props.kernel := rbInvMultqud;
          props.augmentBase := False;
          props.sigma := 0.6;
          props.epsilon := 0.1;
          props.randomCenterPerc := 0.3;
          props.RBFlearnAlgorithm := wlLeastSquares;
          props.centerType := rbRandom;

          learner := TRadialBasisLearner.Create;
          try
             learner.SetProps(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.butNeuralNetClick(Sender: TObject);
var learner : TNeuralNetLearner;
    props : TNeuralNetProps;
begin
     if Assigned(fExamples) then
     begin
          fClassifier.Free;
          FreeAndNil(fClMapBmp);

          props.learnAlgorithm := nnBackpropMomentum;
          props.outputLayer := ntLinear;

          SetLength(props.layers, 2);
          props.layers[0].NumNeurons := 12;
          props.layers[0].NeuronType := ntTanSigmoid;
          props.layers[1].NumNeurons := 8;
          props.layers[1].NeuronType := ntExpSigmoid;
          props.eta := 0.001;
          props.validationDataSetSize := 0.2;
          props.alpha := 0.8;
          props.cf := 0;
          props.normMeanVar := True;
          props.l1Normalization := 0.1;

          props.numMinDeltaErr := 10;
          props.stopOnMinDeltaErr := 1e-4;
          props.maxNumIter := 4000;
          props.minNumIter := 400;

          learner := TNeuralNetLearner.Create;
          try
             learner.SetProps(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;



procedure TfrmClassifierTest.butKMeanClick(Sender: TObject);
var learner : TKMeansLearner;
    props : TKMeansProps;
begin
     if Assigned(fExamples) then
     begin
          fClassifier.Free;
          FreeAndNil(fClMapBmp);

          props.numCluster := 4;
          props.searchPerClass := False;
          props.useMedian := True;
          props.MaxNumIter := 100;
          props.centChangePerc := 0.01;
          props.initCluster := ciKmeansPlusPlus;          

          learner := TKMeansLearner.Create;
          try
             learner.SetProps(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.butLDAClick(Sender: TObject);
var learner : TFisherBatchLDALearner;
    props : TFisherAugmentedBaseProps;
begin
     if Assigned(fExamples) then
     begin
          fClassifier.Free;
          FreeAndNil(fClMapBmp);

          FillChar(Props, sizeof(props), 0);
          props.RobustPCAProps.NumSubSubSpaces := 1;
          props.RobustPCAProps.SubSpaceSizes := 0.8;
          props.RobustPCAProps.SubSpaceCutEPS := 0.99;
          props.RobustPCAProps.Start := 90;
          props.RobustPCAProps.Stop := 80;
          props.RobustPCAProps.ReductionFactor := 0.75;
          
          props.UseFullSpace := True;
          if rbFisherLDA.Checked
          then
              props.ClassifierType := ctFast
          else if rbRobustFisher.Checked
          then
              props.ClassifierType := ctRobust
          else
              props.ClassifierType := ctFastRobust;

          props.NumLDAVectorsToKeep := 1;
          
          learner := TFisherBatchLDALearner.Create;
          try
             learner.SetProperties(props);

             Learner.Init(fExamples);
             fClassifier := Learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.butAdaBoostLoadClick(Sender: TObject);
var haar2DClassifier : THaar2DAdaBoost;
begin
     if odAdaBoost.Execute(Handle) then
     begin
          haar2DClassifier := ReadObjFromFile(odAdaBoost.FileName) as THaar2DAdaBoost;
          try
             // ##################################################
             // #### Load all images and show the classification result
             fSlidingWin := THaar2DSlidingWindow.Create(haar2DClassifier, 2, 2, 0.2, 1, 71/24);
             try
                fSlidingWin.CombineOverlappingReg := True;
                with TIncrementalImageList.Create do
                try
                   OnImageStep := adaBoostImgStep;
                   Recursive := True;

                   ReadListFromDirectory(edFaceDB.Text, ctGrayScale);
                finally
                       Free;
                end;
             finally
                    fSlidingWin.Free;
             end;
          finally
                 pbBoostProgress.Position := 0;
                 haar2DClassifier.Free;
          end;
     end;
end;

procedure TfrmClassifierTest.butC45Click(Sender: TObject);
var learner : TC45Learner;
    props : TC45Props;
begin
     if Assigned(fExamples) then
     begin
          FreeAndNil(fClMapBmp);
          
          FillChar(props, sizeof(props), 0);
          props.LearnType := ltPrune;
          props.ValidationsetSize := 0.0;
          props.UseValidationSet := True;
          props.MaxDepth := 4;

          fClassifier.Free;
          learner := TC45Learner.Create;
          try
             learner.SetProperties(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.butCreateGaussSetClick(Sender: TObject);
const means : Array[0..3] of double = (3.5, 3.5, 4.5, 4.5);
      stdevs : array[0..3] of double = (0.4, 0.4, 0.5, 0.5); //(1, 1.5, 2, 0.5);
      classLabels : array[0..1] of integer = (-1, 1);
begin
     fExamples.Free;
     FreeAndNil(fClassifier);
     FreeAndNil(fClMapBmp);

{$IFDEF INITRANDSEED}
     RandSeed := 100;
{$ENDIF}

     // testing a gaussian distribution
     fexamples := Create2DGaussSet(means, stdevs, classLabels, 2, [50, 50]);
     PaintBox1.Repaint;
end;

procedure TfrmClassifierTest.butDecissionStumpClick(Sender: TObject);
var learner : TDecisionStumpLearner;
begin
     if Assigned(fExamples) then
     begin
          FreeAndNil(fClMapBmp);
          fClassifier.Free;
          learner := TDecisionStumpLearner.Create;
          try
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.butAdaBoostClick(Sender: TObject);
var learner : TDiscreteAdaBoostLearner;
    props : TBoostProperties;
begin
     if Assigned(fExamples) then
     begin
          FreeAndNil(fClMapBmp);
          fClassifier.Free;

          learner := TDiscreteAdaBoostLearner.Create;
          try
             props.NumRounds := 20;
             props.PruneToLowestError := False;
             props.InitClassSpecificWeights := True;
             props.OwnsLearner := True;
             props.WeakLearner := TDecisionStumpLearner.Create;

             learner.SetProperties(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.butGentleBoostClick(Sender: TObject);
var learner : TGentleBoostLearner;
    props : TBoostProperties;
begin
     if Assigned(fExamples) then
     begin
          FreeAndNil(fClMapBmp);
          fClassifier.Free;

          learner := TGentleBoostLearner.Create;
          try
             props.NumRounds := 100;
             props.PruneToLowestError := True;
             props.InitClassSpecificWeights := True;
             props.OwnsLearner := True;
             props.WeakLearner := TDecisionStumpLearner.Create;

             learner.SetProperties(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.butBaggingClick(Sender: TObject);
var learner : TVotedBaggingLearner;
    props : TVotedBaggingProps;
begin
     if Assigned(fExamples) then
     begin
          FreeAndNil(fClMapBmp);
          fClassifier.Free;

          learner := TVotedBaggingLearner.Create;
          try
             props.NumRounds := 100;
             props.balanced := True;
             props.LearnDataSetPercentage := 66;
             props.BalanceOnlyOneTime := True;
             props.Learner := TDecisionStumpLearner.Create;
             props.OwnsLearner := True;

             learner.SetBaggingParams(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.Button6Click(Sender: TObject);
var learner : TFisherBatchLDALearner;
begin
     if Assigned(fExamples) then
     begin
          FreeAndNil(fClMapBmp);
          fClassifier.Free;

          learner := TFisherBatchLDALearner.Create;
          try
             learner.Init(fExamples);
             fClassifier := learner.Learn;
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.butImgRobustFisherLDAClick(Sender: TObject);
var props : TFisherAugmentedBaseProps;
    clProps : TFisherRobustLDAProps;
    examples : TCustomIncrementalLearnerExampleList;
begin
     if not DirectoryExists('.\Faces\') then
     begin
          ShowMessage('Face directory does not exists');
          exit;
     end;
     FreeAndNil(fExamples);
     FreeAndNil(fClassifier);

     props.UseFullSpace := False;
     if rbFisherLDA.Checked
     then
         props.ClassifierType := ctFast
     else if rbRobustFisher.Checked
     then
         props.ClassifierType := ctRobust
     else
         props.ClassifierType := ctFastRobust;

     props.NumLDAVectorsToKeep := 8;
     props.RobustPCAProps.NumSubSubSpaces := 80;
     props.RobustPCAProps.SubSpaceSizes := 0.003;    // use 108 pixel per hypothesis
     props.RobustPCAProps.Start := 50;
     props.RobustPCAProps.Stop := 20;
     props.RobustPCAProps.ReductionFactor := 0.75;
     props.RobustPCAProps.SubSpaceCutEPS := 0.9;

     if Sender = butImgRobustFisherLDA then
     begin
          fExamples := TImageMatrixExampleList.Create('.\Faces\', ctGrayScale);

          with TFisherBatchLDAAugmentedBaseLearner.Create do
          try
             SetProperties(props);

             Init(fExamples);
             fClassifier := Learn;
          finally
                 Free;
          end;
     end
     else
     begin
          Examples := TIncrementalImageExampleList.Create('.\Faces\', ctGrayScale, 0.2, lsOneByOne);
          try
             with TFisherIncrementalLDA.Create do
             try
                SetProperties(props);

                Init(Examples);
                fClassifier := Learn;
             finally
                    Free;
             end;
          finally
                 Examples.Free;
          end;

          // for all other procedures (learn error) we need the complete list:
          fExamples := TImageMatrixExampleList.Create('.\Faces\', ctGrayScale);
     end;

     if fClassifier is TFisherRobustLDAClassifier then
     begin
          clProps.NumHypothesis := 13;
          clProps.Start := 50;
          clProps.Stop := 20;
          clProps.ReductionFactor := 0.75;
          clProps.K2 := 0.01;
          clProps.accurFit := True;
          clProps.maxIter := 3;
          clProps.theta := 60;

          TFisherRobustLDAClassifier(fClassifier).SetProps(clProps);
     end;

     TestUnseenImages;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.butSVMClick(Sender: TObject);
var learner : TSVMLearner;
    props : TSVMProps;
begin
     InitMathFunctions(itFPU, False);
     if Assigned(fExamples) then
     begin
          fClassifier.Free;
          FreeAndNil(fClMapBmp);

          props.learnMethod := lmLagrangian;
          props.autoScale := true;
          //props.kernelType := svmGauss;
          //props.sigma := 0.51;
          //props.kernelType := svmSigmoid;
//          props.scale := 0.5;
//          props.offset := 1;

          props.kernelType := svmPolyInhomogen;
          props.order := 3;
          props.slack := 1;

          learner := TSVMLearner.Create;
          try
             learner.SetProps(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn(Weights);
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

function TfrmClassifierTest.Create2DGaussSet(const mean,
  stddev: array of Double; const classLabels : Array of integer; numDim : integer; const numExamples : Array of integer): TCustomLearnerExampleList;
var i, j, k : integer;
    matrix : TDoubleMatrix;
    classvals : Array of integer;
    actClassIdx : integer;
begin
     assert(Length(mean) = length(stddev), 'Dimension error');
     assert(Length(mean) mod numDim = 0, 'Dimension error');
     assert(Length(classLabels) = Length(stddev) div numDim, 'Dimension error');
     Setlength(classvals, SumInt( numExamples ));

     // this is a simple test matrix for easier debugging
     //matrix := TDoubleMatrix.Create(4, 2);
//     matrix[0, 0] := -1;
//     matrix[0, 1] := -0.9;
//     matrix[1, 0] := -0.4;
//     matrix[1, 1] := -0.5;
//     matrix[2, 0] := 1;
//     matrix[2, 1] := 0.8;
//     matrix[3, 0] := 0.9;
//     matrix[3, 1] := 0.4;
//     //matrix[4, 0] := 0.7;
////     matrix[4, 1] := -0.1;
//
//     SetLength(classVals, 4);
//     classVals[0] := -1;
//     classVals[1] := -1;
//     classVals[2] := 1;
//     classVals[3] := 1;
//
//     Result := TMatrixLearnerExampleList.Create(matrix, classvals, True);
//
//     fMinVal := -1;
//     fMaxVal := 1;
//     exit;

     fMinVal := MaxDouble;
     fMaxVal := -MaxDouble;

     matrix := TDoubleMatrix.Create(Length(classVals), numDim);
     actClassIdx := 0;
     for i := 0 to Length(stddev) div numDim - 1 do
     begin
          for j := 0 to numExamples[i] - 1 do
          begin
               for k := 0 to numDim - 1 do
               begin
                    matrix[actClassIdx, k] := RandG(mean[i*numDim + k], stddev[i*numDim + k]);

                    fMinVal := Min(fMinVal, matrix[actClassIdx, k]);
                    fMaxVal := Max(fMaxVal, matrix[actClassIdx, k]);
               end;

               classvals[actClassIdx] := classLabels[i];
               inc(actClassIdx);
          end;
     end;

    // with TStringList.Create do
//     try
//        DecimalSeparator := '.';
//        beginUpdate;
//        for i := 0 to matrix.width - 1 do
//        begin
//             Add( Format('%.6f, %.6f, %d', [matrix[i, 0], matrix[i, 1], classVals[i] ] ));
//        end;
//        EndUpdate;
//        SaveToFile('D:\testcl.txt', TEncoding.ASCII);
//     finally
//            Free;
//     end;
     
     Result := TMatrixLearnerExampleList.Create(matrix, classvals, True);
end;

procedure TfrmClassifierTest.FormCreate(Sender: TObject);
begin
     ReportMemoryLeaksOnShutdown := True;
end;

procedure TfrmClassifierTest.FormDestroy(Sender: TObject);
begin
     fExamples.Free;
     fClassifier.Free;
     fClMapBmp.Free;

     fFace1.Free;
     fFace2.Free;
     fFaceTest.Free;
     fFaceReconstruct.Free;
end;

procedure TfrmClassifierTest.MergeNegExamples(const exmplFileName: string;
  lst: TClassRecList);
var posLst : TClassRecList;
    sl1, sl2 : TStringList;
    cnt, i: Integer;
    x1, y1, x2, y2 : integer;
    overlapped : boolean;
    rc1, rc2, rc3 : TRect;
    w1, h1 : integer;
begin
     // ######################################################
     // #### Read positive example list
     posLst := TClassRecList.Create;

     sl1 := TStringList.Create;
     sl2 := TStringList.Create;
     try
        sl1.LoadFromFile(ChangeFileExt(exmplFileName, '.pos'));
        for cnt := 0 to sl1.Count - 1 do
        begin
             sl2.CommaText := sl1[cnt];

             x1 := StrToInt(sl2[0]);
             x2 := StrToInt(sl2[2]);
             y1 := StrToInt(sl2[1]);
             y2 := StrToInt(sl2[3]);
             posLst.Add(TClassRec.Create(x1, y1, x2 - x1 + 1, y2 - y1 + 1));
        end;


        sl1.Clear;
        if FileExists(ChangeFileExt(exmplFileName, '.neg')) then
           sl1.LoadFromFile(ChangeFileExt(exmplFileName, '.neg'));

        // add examples to negative list if the overlapping area is lower than 80%
        for cnt := 0 to Min(50, lst.Count) - 1 do
        begin
             // check for overlap
             overlapped := False;
             for i := 0 to posLst.Count - 1 do
             begin
                  rc1 := Rect(posLst[i].x, posLst[i].y, posLst[i].Right, posLst[i].Bottom);
                  rc2 := Rect(lst[i].x, lst[i].y, lst[i].Right, lst[i].Bottom);

                  IntersectRect(rc3, rc1, rc2);

                  w1 := rc3.Right - rc3.Left + 1;
                  h1 := rc3.Bottom - rc3.Top + 1;

                  if w1*h1 > 0.5*(rc2.Right - rc1.Left + 1)*(rc1.Bottom - rc1.Top + 1) then
                  begin
                       overlapped := True;
                       break;
                  end;
             end;

             if not overlapped then
                sl1.Add(IntToStr(lst[cnt].x) + ',' + IntToStr(lst[cnt].y) + ',' +
                        IntToStr(lst[cnt].Right) + ',' + IntToStr(lst[cnt].Bottom));
        end;

        sl1.SaveToFile(ChangeFileExt(exmplFileName, '.neg'));
     finally
            posLst.Free;
            sl1.Free;
            sl2.Free;
     end;
end;

procedure TfrmClassifierTest.OnPCAReconstruct(Sender: TObject; rec: TDoubleMatrix);
begin
     if not Assigned(fFaceReconstruct) and Assigned(fExamples) then
     begin
          with TMatrixImageConverter.Create(ctGrayScale, False, False, TImageMatrixExampleList(fExamples).ImgWidth,
                                            TImageMatrixExampleList(fExamples).ImgHeight) do
          try
             fFaceReconstruct := MatrixToImage(rec);
          finally
                 Free;
          end;
     end;
end;

type
  TC45CLHack = class(TC45Classifier);

procedure TfrmClassifierTest.PaintBox1Paint(Sender: TObject);
const cColors : Array[-1..1] of TColor = (clRed, clWhite, clBlue);
var i : integer;
    x, y : integer;
    xDim : double;
    yDim : double;
    stump : TDecisionStump;
    thresh : double;
    minXVal, maxXVal : double;
    minYVal, maxYVal : double;
    ensemble : TEnsembelClassifier;
    conf : double;
    cl : TCustomClassifier;
    centers : TDoubleMatrixDynArr;
    mean : TDoubleMatrix;
    direction : TDoubleMatrix;
procedure PaintStump(stump : TDecisionStump);
begin
     thresh := stump.Threshold;
     PaintBox1.Canvas.Pen.Color := clBlack;
     if stump.Dimension = 1 then
     begin
          y := Trunc(PaintBox1.Height - (thresh - minYVal)*yDim);

          PaintBox1.Canvas.MoveTo(0, y);
          PaintBox1.Canvas.LineTo(PaintBox1.Width, y);
     end
     else
     begin
          x := Trunc(((Thresh - minXVal)*xDim));

          PaintBox1.Canvas.MoveTo(x, 0);
          PaintBox1.Canvas.LineTo(x, PaintBox1.Height);
     end;
end;

function GetLeftClassFromNode(tree : TCustomTreeItem) : integer;
begin
     if tree is TTreeNode
     then
         Result := GetLeftClassFromNode(TTreeNode(tree).LeftItem)
     else
         Result := T45NodeData(TTreeLeave(tree).TreeData).Classes[0];
end;

procedure PaintC45Stub(tree : TCustomTreeItem; iter : integer; isleft : boolean);
begin
     if tree is TTreeNode then
     begin
          thresh := T45NodeData(tree.TreeData).SplitVal;

          if T45NodeData(tree.TreeData).FeatureSplitIndex = 1 then
          begin
               y := Trunc(PaintBox1.Height - (thresh - minYVal)*yDim);

               //PaintBox1.Canvas.Pen.Color := ifthen(leftCl = 0, clBlue, clRed);
               PaintBox1.Canvas.Pen.Color := clGray; // + RGB(ifthen(isleft, 0, iter*30), ifthen(isleft, iter*30, 0), iter*30);
               PaintBox1.Canvas.MoveTo(0, y);
               PaintBox1.Canvas.LineTo(PaintBox1.Width, y);
          end
          else
          begin
               x := Trunc(((Thresh - minXVal)*xDim));

               //PaintBox1.Canvas.Pen.Color := ifthen(leftCl = 0, clBlue, clRed);
               PaintBox1.Canvas.Pen.Color := clGray; //RGB(ifthen(isleft, 0, iter*30), ifthen(isleft, iter*30, 0), iter*30);
               PaintBox1.Canvas.MoveTo(x, 0);
               PaintBox1.Canvas.LineTo(x, PaintBox1.Height);
          end;

          PaintC45Stub(TTreeNode(tree).LeftItem, iter + 1, True);
          PaintC45Stub(TTreeNode(tree).RightItem, iter + 1, False);
     end
     else
     begin
          thresh := T45NodeData(tree.TreeData).SplitVal;
          if T45NodeData(tree.TreeData).FeatureSplitIndex = 1 then
          begin
               y := Trunc(PaintBox1.Height - (thresh - minYVal)*yDim);

               //PaintBox1.Canvas.Pen.Color := ifthen(leftCl = 0, clBlue, clRed);
               PaintBox1.Canvas.Pen.Color := clBlack;// + RGB(ifthen(isleft, 0, iter*30), ifthen(isleft, iter*30, 0), iter*30);
               PaintBox1.Canvas.MoveTo(0, y);
               PaintBox1.Canvas.LineTo(PaintBox1.Width, y);
          end
          else
          begin
               x := Trunc(((Thresh - minXVal)*xDim));

               //PaintBox1.Canvas.Pen.Color := ifthen(leftCl = 0, clBlue, clRed);
               PaintBox1.Canvas.Pen.Color := clBlack;// + RGB(ifthen(isleft, 0, iter*30), ifthen(isleft, iter*30, 0), iter*30);
               PaintBox1.Canvas.MoveTo(x, 0);
               PaintBox1.Canvas.LineTo(x, PaintBox1.Height);
          end;
     end;
end;
procedure PaintC45(cl : TC45CLHack);
var actTree : TCustomTreeItem;
begin
     actTree := cl.Tree;
     PaintC45Stub(actTree, 0, False);
end;

procedure PaintRBFCenters(cl : TRBFClassifier);
var aMtx : IMatrix;
    cnt: Integer;
begin
     aMtx := cl.Centers.Clone;

     PaintBox1.Canvas.Brush.Style := bsClear;
     for cnt := 0 to aMtx.Width - 1 do
     begin
          x := Trunc(((aMtx[cnt, 0] - minXVal)*xDim));
          y := Trunc(PaintBox1.Height - (aMtx[cnt, 1] - minYVal)*yDim);

          assert((x >= 0) and (x <= PaintBox1.Width), 'Error x out of bounds');
          assert((y >= 0) and (y <= PaintBox1.Height), 'Error y out of bounds');

          PaintBox1.Canvas.Pen.Color := clMaroon;
          PaintBox1.Canvas.Ellipse(x - 6, y - 6, x + 6, y + 6);
     end;
end;

procedure PaintKMeansCenters(cl : TKMeans);
var aMtx : IMatrix;
    cnt: Integer;
begin
     aMtx := cl.Centers.Clone;

     PaintBox1.Canvas.Brush.Style := bsClear;
     for cnt := 0 to aMtx.Width - 1 do
     begin
          x := Trunc(((aMtx[cnt, 0] - minXVal)*xDim));
          y := Trunc(PaintBox1.Height - (aMtx[cnt, 1] - minYVal)*yDim);

          PaintBox1.Canvas.Pen.Color := clMaroon;
          PaintBox1.Canvas.Ellipse(x - 6, y - 6, x + 6, y + 6);
     end;
end;

procedure PaintCLBoundary(cl : TCustomClassifier; polyLine : boolean);
var map : Array of Array of integer;
    conf : Array of Array of double;
    aMtx : TDoubleMatrix;
    xmpl : TMatrixExample;
    x, y : integer;
    pts : Array of TPoint;
    numPts : integer;
const divider : integer = 4;
procedure OrderPts;
var i, j : integer;
    hlp : TPoint;
    minIdx : integer;
    minDist : double;
    actDist : double;
    sortIdx : integer;
begin
     for i := 0 to numPts - 2 do
     begin
          minIdx := i + 1;
          minDist := sqr(pts[i].X - pts[i + 1].X) + sqr(pts[i].Y - pts[i + 1].Y);
          sortIdx := i + 2;
          for j := i + 2 to numPts - 1 do
          begin
               actDist := sqr(pts[i].X - pts[j].X) + sqr(pts[i].Y - pts[j].Y);
               if minDist > actDist then
               begin
                    minIdx := j;
                    minDist := actDist;
               end
               else if minDist = actDist then
               begin
                    // make it the next point to proceed
                    hlp := pts[j];
                    pts[j] := pts[sortIdx];
                    pts[sortIdx] := hlp;
                    inc(sortIdx);
               end;
          end;

          hlp := pts[i + 1];
          pts[i + 1] := pts[minIdx];
          pts[minIdx] := hlp; 
     end;
end;
function ColorizeConf(color : TColor; conf : double) : TColor;
begin
     // we have red and blue
     Result := color or
                 Integer( RGB( Trunc($7f + Max(0, Min(1, 1 - conf))*$80),
                                                  Trunc($7f + Max(0, Min(1, 1 - conf))*$80),
                                                  Trunc($7f + Max(0, Min(1, 1 - conf))*$80)
                                                    ));

     //Result := color or
//                 Integer( RGB( Trunc($0 + Max(0, Min(1, 1 - conf))*$FF),
//                                                  Trunc($0 + Max(0, Min(1, 1 - conf))*$FF),
//                                                  Trunc($0 + Max(0, Min(1, 1 - conf))*$FF)
//                                                    ));
end;
var freq, start, stop : Int64;
begin
     if not Assigned(fClMapBmp) then
     begin
          QueryPerformanceFrequency(freq);
          QueryPerformanceCounter(start);
          fClMapBmp := TBitmap.Create;
          fClMapBmp.SetSize(PaintBox1.Width, PaintBox1.Height);

          // classify each pixel and set it's color if neighbouring pixels are different
          SetLength(map, PaintBox1.Height div divider, PaintBox1.Width div divider);
          SetLength(conf, PaintBox1.Height div divider, PaintBox1.Width div divider);

          aMtx := TDoubleMatrix.Create(1, 2);
          xmpl := TMatrixExample.Create(aMtx, false);

          for y := 0 to PaintBox1.Height div divider - 1 do
          begin
               for x := 0 to PaintBox1.Width div divider - 1 do
               begin
                    aMtx.Vec[0] := minXVal + divider*x/xDim;
                    aMtx.Vec[1] := minYVal + divider*y/yDim;
                    
                    map[PaintBox1.Height div divider - y - 1][x] := cl.Classify(xmpl, conf[PaintBox1.Height div divider - y - 1][x]);
               end;
          end;
          xmpl.Free;
          aMtx.Free;

          QueryPerformanceCounter(stop);
          OutputDebugString( PChar( Format('Confidence map took %.3fms', [ 1000*(stop - start)/freq ] ) ) );

          fClMapBmp.Canvas.Brush.Color := PaintBox1.Color;
          fClMapBmp.Canvas.FillRect(Rect(0, 0, fClMapBmp.Width, fClMapBmp.Height));

          if chkConfidence.Checked then
          begin
               fClMapBmp.canvas.Brush.Style := bsSolid;

               for y := 0 to Length(conf) - 1 do
               begin
                    for x := 0 to Length(conf[0]) - 1 do
                    begin
                         if conf[y][x] <> 0 then
                         begin
                              fClMapBmp.Canvas.Brush.Color := ColorizeConf(cColors[map[y][x]], conf[y][x]);

                              fClMapBmp.Canvas.FillRect(Rect(x*divider,
                                                            y*divider,
                                                            x*divider + divider + 1,
                                                            y*divider + divider + 1));
                         end;
                    end;
               end;
          end;

          fClMapBmp.Canvas.Brush.Color := clDkGray;
          fClMapBmp.canvas.Brush.Style := bsSolid;

          // check in x direction
          if not polyLine then
          begin
               for y := 1 to Length(map) - 1 do
               begin
                    for x := 1 to Length(map[0]) - 1 do
                        if (map[y][x-1] <> map[y][x]) or
                           (map[y-1][x] <> map[y][x]) then
                        begin
                             fClMapBmp.Canvas.FillRect(Rect(x*divider - 1,
                                                            y*divider - 1,
                                                            x*divider + 2,
                                                            y*divider + 2));
                        end;
               end;
          end
          else
          begin
               SetLength(pts, (PaintBox1.Height div divider)*(PaintBox1.Width div divider));

               numPts := 0;
               // check in x direction
               for y := 1 to Length(map) - 1 do
               begin
                    for x := 1 to Length(map[0]) - 1 do
                        if ((map[y][x-1] <> map[y][x])) or
                           ((map[y-1][x] <> map[y][x])) then
                        begin
                             pts[numPts].X := x*divider;
                             pts[numPts].Y := y*divider;
                             inc(numPts);
                        end;
               end;

               OrderPts;

               fClMapBmp.Canvas.Pen.Color := clDkGray;
               fClMapBmp.Canvas.Polyline(Copy(pts, 0, numPts));
          end;
     end;

     PaintBox1.Canvas.StretchDraw(PaintBox1.ClientRect, fClMapBmp);
end;

begin
     if not Assigned(fExamples) then
        exit;

     if fExamples is TImageMatrixExampleList then
     begin
          PaintFaceClassifier;
          exit;
     end;

     // get dimension factor -> fit all features
     minXVal := 1e6;
     maxXVal := -1e6;
     minYVal := 1e6;
     maxYVal := -1e6;

     for i := 0 to fExamples.Count - 1 do
     begin
          minXVal := Min(minXVal, fExamples[i].FeatureVec[0]);
          maxXVal := Max(maxXVal, fExamples[i].FeatureVec[0]);
          minYVal := Min(minYVal, fExamples[i].FeatureVec[1]);
          maxYVal := Max(maxYVal, fExamples[i].FeatureVec[1]);
     end;


     xDim := PaintBox1.Width/(1.1*(maxXval - minXVal));
     yDim := PaintBox1.Height/(1.1*(maxYVal - minYVal));
     minXVal := minXVal - 0.05*(maxXVal - minXVal);
     minYVal := minYVal - 0.05*(maxYVal - minYVal);

     if Assigned(fClassifier) then
        PaintCLBoundary(fClassifier, not ((fClassifier is TRBFClassifier) or (fClassifier is TC45Classifier)));

     if Assigned(fClassifier) and (fClassifier is TRBFClassifier) then
        PaintRBFCenters(TRBFClassifier(fClassifier));

     if Assigned(fClassifier) and (fClassifier is TKMeans) then
        PaintKMeansCenters(TKMeans( fClassifier ) );

     PaintBox1.Canvas.Brush.Style := bsSolid;

     if Assigned(fClassifier) and (fClassifier is TDecisionStump) then
     begin
          stump := fClassifier as TDecisionStump;

          PaintStump(stump);
     end;
     if Assigned(fClassifier) then
     begin
          cl := fClassifier as TCustomClassifier;

          PaintBox1.Canvas.Brush.Style := bsClear;
          for i := 0 to fExamples.Count - 1 do
          begin
               x := Trunc(((fExamples[i].FeatureVec[0] - minXVal)*xDim));
               y := Trunc(PaintBox1.Height - (fExamples[i].FeatureVec[1] - minYVal)*yDim);

               PaintBox1.Canvas.Pen.Color := cColors[cl.Classify(fExamples[i], conf)];
               PaintBox1.Canvas.Ellipse(x - 4, y - 4, x + 4, y + 4);
          end;

          if fClassifier is TEnsembelClassifier then
          begin
               ensemble := fClassifier as TEnsembelClassifier;
               if (ensemble.Classifiers.Count > 0) and (ensemble.Classifiers.Classifiers[0] is TDecisionStump) then
               begin
                    for i := 0 to ensemble.Classifiers.Count - 1 do
                        PaintStump(ensemble.Classifiers[i] as TDecisionStump);
               end;
          end;

          if fClassifier is TC45Classifier then
             PaintC45(TC45CLHack(fClassifier));

          if cl is TFisherLDAClassifier then
          begin
               PaintBox1.Canvas.Pen.Color := clMaroon;
               PaintBox1.Canvas.Brush.Style := bsClear;

               centers := TFisherLDAClassifier(cl).BackProjectedCenters;
               try
                  for i := 0 to length(centers) - 1 do
                  begin
                       x := Trunc(((centers[i][0, 0] - minXVal)*xDim));
                       y := Trunc(PaintBox1.Height - (centers[i][0, 1] - minYVal)*yDim);

                       PaintBox1.Canvas.Ellipse(x - 6, y - 6, x + 6, y + 6);

                       if i < Length(centers) - 1 then
                       begin
                            mean := centers[i].Add(centers[i + 1]);
                            try
                               mean.ScaleInPlace(0.5);

                               direction := centers[i].Sub(centers[i + 1]);
                               try
                                  conf := direction[0, 0];
                                  direction[0, 0] := direction[0, 1];
                                  direction[0, 1] := conf;
                                  direction[0, 1] := -direction[0, 1];
                                  direction.ScaleInPlace(1/sqrt(sqr(direction[0, 0]) + sqr(direction[0, 1])));

                                  x := Trunc((mean[0, 0] - PaintBox1.Width*direction[0, 0] - minXVal)*xDim);
                                  y := PaintBox1.Height - Trunc((mean[0, 1] - PaintBox1.Width*direction[0, 1] - minYVal)*yDim);
                                  PaintBox1.Canvas.MoveTo(x, y);

                                  x := Trunc((mean[0, 0] + PaintBox1.Width*direction[0, 0] - minXVal)*xDim);
                                  y := PaintBox1.Height - Trunc((mean[0, 1] + PaintBox1.Width*direction[0, 1] - minYVal)*yDim);
                                  PaintBox1.Canvas.LineTo(x, y);
                               finally
                                      direction.Free;
                               end;
                            finally
                                   mean.Free;
                            end;
                       end;
                  end;
               finally
                      for i := 0 to Length(centers) - 1 do
                          centers[i].Free;
               end;
          end;
     end;

     // paint for each example one dot
     for i := 0 to fExamples.Count - 1 do
     begin
          x := Trunc(((fExamples[i].FeatureVec[0] - minXVal)*xDim));
          y := Trunc(PaintBox1.Height - (fExamples[i].FeatureVec[1] - minYVal)*yDim);

          assert((x >= 0) and (x <= PaintBox1.Width), 'Error x out of bounds');
          assert((y >= 0) and (y <= PaintBox1.Height), 'Error y out of bounds');

          PaintBox1.Canvas.Pen.Color := cColors[fExamples[i].ClassVal];
          PaintBox1.Canvas.Brush.Color := cColors[fExamples[i].ClassVal];
          PaintBox1.Canvas.Ellipse(x - 2, y - 2, x + 2, y + 2);
     end;
end;

procedure TfrmClassifierTest.PaintFaceClassifier;
var faces : TDoubleMatrixDynArr;
begin
     // #################################################
     // #### Paint backprojected face class centers
     if (( (fClassifier is TFisherLDAClassifier) ) or ((fClassifier is TIncrementalFisherLDAClassifier))) and not Assigned(fFace1) then
     begin
          if (fClassifier is TFisherLDAClassifier) 
          then
              faces := (fClassifier as TFisherLDAClassifier).BackProjectedCenters
          else
              faces := (fClassifier as TIncrementalFisherLDAClassifier).BackProjectedCenters;

          assert(Length(faces) = 2, 'Error only 2 faces expected');

          with TMatrixImageConverter.Create(ctGrayScale, False, False, (fExamples as TImageMatrixExampleList).ImgWidth,
                                 (fExamples as TImageMatrixExampleList).ImgHeight) do
          try
             fFace1 := MatrixToImage(faces[0]);
             fFace2 := MatrixToImage(faces[1]);
          finally
                 Free;
          end;

          faces[0].Free;
          faces[1].Free;
     end;

     if Assigned(fFace1) then
        PaintBox1.Canvas.StretchDraw(Rect(0, 0, PaintBox1.Width div 2, PaintBox1.Height div 2), fFace1);
     if Assigned(fFace2) then
        PaintBox1.Canvas.StretchDraw(Rect(PaintBox1.Width div 2, 0, PaintBox1.Width, PaintBox1.Height div 2), fFace2);
     if Assigned(fFaceReconstruct) then
        PaintBox1.Canvas.StretchDraw(Rect(0, PaintBox1.Height div 2, PaintBox1.Width div 2, PaintBox1.Height), fFaceReconstruct);
     if Assigned(fFaceTest) then
        PaintBox1.Canvas.StretchDraw(Rect(PaintBox1.Width div 2, PaintBox1.Height div 2, PaintBox1.Width, PaintBox1.Height), fFaceTest);

end;

procedure TfrmClassifierTest.TestLearnError;
var numCorrectClassified : integer;
    i : Integer;
    conf : double;
    start, Stop : Cardinal;
begin
     lblLearnError.Caption := '0';

     if Assigned(fClassifier) then
     begin
          start := GetTickCount;
          numCorrectClassified := 0;

          for i := 0 to fExamples.Count - 1 do
          begin
               if fExamples[i].ClassVal = fClassifier.Classify(fExamples[i], conf) then
                  inc(numCorrectClassified);
          end;
          stop := GetTickCount;

          lblLearnError.Caption := Format('%d of %d, %.2f %% - Time: %d', [numCorrectClassified, fExamples.Count, numCorrectClassified/fExamples.Count*100, stop - start]);
     end;
end;

procedure TfrmClassifierTest.TestUnseenImages;
var exmpl : TCustomExample;
    img : TDoubleMatrix;
    pict : TPicture;
    classVal : integer;
    i : integer;
    s : string;
    labels : TStringList;
    bmp : TBitmap;
    rc : TRect;
begin
     // test the classifier with an unknown image
     FreeAndNil(fFaceTest);
     FreeAndNil(fFaceReconstruct);

     labels := TStringList.Create;
     try
        labels.LoadFromFile('.\faces\labels.txt');
        lblOrigLabels.Caption := labels.CommaText;
        labels.Clear;

        i := 1;
        s := Format('.\faces\Face%d_TestImg.jpg', [i]);
        while FileExists(s) do
        begin
             pict := TPicture.Create;
             try
                pict.LoadFromFile(s);
                bmp := TBitmap.Create;

                try
                   bmp.SetSize(pict.Width, pict.Height);
                   bmp.Canvas.Draw(0, 0, pict.Graphic);

                   // ################################################
                   // #### paint a black bar into the facial image
                   // covering 20% of the face
                   if chkBlendPart.Checked then
                   begin
                        bmp.Canvas.Brush.Color := clWhite;

                        rc.Left := bmp.Width div 2;
                        rc.top := 0;
                        rc.Right := rc.Left + Trunc(bmp.Width*0.25);
                        rc.Bottom := bmp.Height;
                        bmp.Canvas.FillRect(rc);
                   end;

                   with TMatrixImageConverter.Create(ctGrayScale, False, False, bmp.Width, bmp.Height) do
                   try
                      img := ImageToMatrix(bmp);
                   finally
                          Free;
                   end;

                   if not Assigned(fFaceTest) then
                   begin
                        fFaceTest := TBitmap.Create;
                        fFaceTest.Assign(bmp);
                   end;
                finally
                       bmp.Free;
                end;

                exmpl := TMatrixExample.Create(img, True);
                try
                   if fClassifier is TFisherLDAClassifier then
                      TFisherLDAClassifier(fClassifier).OnReconstructFace := OnPCAReconstruct;

                   if fClassifier is TIncrementalFisherLDAClassifier then
                      TIncrementalFisherLDAClassifier(fClassifier).OnReconstructFace := OnPCAReconstruct;
                   
                   classVal := fClassifier.Classify(exmpl);

                   labels.Add(IntToStr(classVal));
                finally
                       exmpl.Free;
                end;
             finally
                    pict.Free;
             end;

             inc(i);
             s := Format('.\faces\Face%d_TestImg.jpg', [i]);
        end;

        if fClassifier is TFisherLDAClassifier then
           TFisherLDAClassifier(fClassifier).OnReconstructFace := nil;
        lblUnseen.Caption := labels.CommaText;
     finally
            labels.Free;
     end;
end;

procedure TfrmClassifierTest.chkConfidenceClick(Sender: TObject);
begin
     FreeAndNil(fClMapBmp);

     PaintBox1.Invalidate;
end;

function TfrmClassifierTest.Weights: TDoubleDynArray;
var sum : double;
  counter: Integer;
begin
     SetLength(Result, fExamples.Count);

     if chkWeights.Checked then
     begin
          sum := 0;
          for counter := 0 to Length(Result) - 1 do
          begin
               Result[counter] := random;
               sum := sum + Result[counter];
          end;

          for counter := 0 to Length(Result) - 1 do
              Result[counter] := Result[counter]/sum;
     end
     else
     begin
          for counter := 0 to Length(Result) - 1 do
              Result[counter] := 1/Length(Result);
     end;

end;

end.
