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

{$DEFINE INITRANDSEED}  // uncomment if you do not want the same train set 

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, BaseClassifier, ExtCtrls, StdCtrls, Matrix, ComCtrls, Haar2DAdaBoost,
  Haar2DImageSweep, Image2DSweep;

type
  TfrmClassifierTest = class(TForm)
    GroupBox1: TGroupBox;
    Button1: TButton;
    PaintBox1: TPaintBox;
    Button2: TButton;
    Button3: TButton;
    Label1: TLabel;
    lblLearnError: TLabel;
    Button4: TButton;
    Button5: TButton;
    Button6: TButton;
    GroupBox2: TGroupBox;
    Button7: TButton;
    Label2: TLabel;
    lblUnseen: TLabel;
    lblOrigLabels: TLabel;
    Label3: TLabel;
    chkBlendPart: TCheckBox;
    rbFisherLDA: TRadioButton;
    rbRobustFischer: TRadioButton;
    rbFastRobustFisher: TRadioButton;
    Button8: TButton;
    Button9: TButton;
    Button10: TButton;
    edFaceDB: TEdit;
    Label4: TLabel;
    pbBoostProgress: TProgressBar;
    btnFaceDb: TButton;
    chkSaveClassifier: TCheckBox;
    sdSaveAdaBoost: TSaveDialog;
    butAdaBoostLoad: TButton;
    odAdaBoost: TOpenDialog;
    butC45: TButton;
    btnNaiveBayes: TButton;
    chkAutoMerge: TCheckBox;
    procedure Button1Click(Sender: TObject);
    procedure PaintBox1Paint(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Button3Click(Sender: TObject);
    procedure Button4Click(Sender: TObject);
    procedure Button5Click(Sender: TObject);
    procedure Button6Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure Button7Click(Sender: TObject);
    procedure Button9Click(Sender: TObject);
    procedure Button10Click(Sender: TObject);
    procedure btnFaceBoostingClick(Sender: TObject);
    procedure butAdaBoostLoadClick(Sender: TObject);
    procedure butC45Click(Sender: TObject);
    procedure btnNaiveBayesClick(Sender: TObject);
    procedure Button11Click(Sender: TObject);
  private
    { Private-Deklarationen }
    fFace1, fFace2 : TBitmap;
    fFaceTest : TBitmap;
    fFaceReconstruct : TBitmap;
    fSVMBmp : TBitmap;

    fMinVal, fMaxVal : double;

    fExamples : TCustomLearnerExampleList;
    fClassifier : TCustomClassifier;

    // temporary face classifier
    fSlidingWin : THaar2DSlidingWindow;

    procedure MergeNegExamples(const exmplFileName : string; lst : TClassRecList);
    procedure adaBoostImgStep(Sender : TObject; mtx : TDoubleMatrix; actNum, NumImags : integer; const FileName : string);

    procedure boostLearnProgress(Sender : TObject; progress : integer);
    procedure OnPCAReconstruct(Sender : TObject; rec : TDoubleMatrix);
    procedure PaintFaceClassifier;
    procedure TestLearnError;
    procedure TestUnseenImages;
    function Create2DGaussSet(const mean, stddev : Array of Double; const classLabels : Array of integer;
                              numDim : integer; numExamples : integer) : TCustomLearnerExampleList;
  public
    { Public-Deklarationen }
  end;

var
  frmClassifierTest: TfrmClassifierTest;

implementation

uses BaseMatrixExamples, math, mathutilfunc, SimpleDecisionStump, AdaBoost,
     CustomBooster, Bagging, EnsembleClassifier, FischerBatchLDA, FischerClassifiers,
     ImageDataSet, ImageMatrixConv, jpeg, IncrementalImageDataSet,
     IncrementalFischerLDA, FischerIncrementalClassifiers, BaseIncrementalLearner,
     IntegralImg, Haar2DDataSet, MatrixImageLists, BinaryReaderWriter,
     BaseMathPersistence, DecisionTree45, TreeStructs, NaiveBayes, SVM;

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
     learnParams.baseProps.NumRounds := 20;
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
     finally
            haarLearner.Free;
            ds.Free;
     end;

     // save the classifier
     if chkSaveClassifier.Checked then
     begin
          if sdSaveAdaBoost.Execute(Handle) then
             haar2DClassifier.SaveToFile(sdSaveAdaBoost.FileName, TBinaryReaderWriter);
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

procedure TfrmClassifierTest.btnNaiveBayesClick(Sender: TObject);
var learner : TNaiveBayesLearner;
    props : TNaiveBayesProps;
begin
     if Assigned(fExamples) then
     begin
          fClassifier.Free;

          props.HistoMin := fMinVal;
          props.HistoMax := fMaxVal;

          props.NumBins := 10;

          learner := TNaiveBayesLearner.Create;
          try
             learner.SetProps(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn;
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;


procedure TfrmClassifierTest.Button10Click(Sender: TObject);
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
     pic.LoadFromFile('.\images\phughe.1.jpg');
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

procedure TfrmClassifierTest.Button11Click(Sender: TObject);
begin

end;
{
var cl : TSVMClassifier;
    feature : TDoubleMatrix;
    origCl : TStringList;
    trainData : TStringList;
    counter: Integer;
    line : TStringList;

    exmpl : TMatrixExample;
    ft : TFormatSettings;
    i: Integer;
    numCorrect : integer;
begin
     GetLocaleFormatSettings(0, ft);
     ft.DecimalSeparator := '.';
     cl := SVMClassifier;

     origCl := TStringList.Create;
     origCl.LoadFromFile('D:\Daten\NoiseDetect\clData.txt');

     trainData := TStringList.Create;
     trainData.LoadFromFile('D:\Daten\NoiseDetect\trainData.txt');
     line := TStringList.Create;

     feature := nil;     
     numCorrect := 0;
     for counter := 0 to trainData.Count - 1 do
     begin
          line.CommaText := trainData[counter];

          if feature = nil then
          begin
               feature := TDoubleMatrix.Create(1, line.Count);
               exmpl := TMatrixExample.Create(feature, True);
          end;

          for i := 0 to line.Count - 1 do
              feature[0, i] := StrToFloat(line[i], ft);

          if StrToInt(origCl[counter]) = cl.Classify(exmpl) then
             inc(numCorrect);
     end;

     ShowMessage(Format('%d/%d correctly classified', [numCorrect, trainData.Count]));
     
     exmpl.Free;
     line.Free;
     origCl.Free;
     cl.Free;
end;
}

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
          props.LearnType := ltPrune;
          props.ValidationsetSize := 0.0;
          props.UseValidationSet := True;
          props.MaxDepth := 4;

          fClassifier.Free;
          learner := TC45Learner.Create;
          try
             learner.SetProperties(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn;
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.Button1Click(Sender: TObject);
const means : Array[0..3] of double = (3.5, 3.5, 4.5, 4.5);
      stdevs : array[0..3] of double = (0.4, 0.4, 0.5, 0.5); //(1, 1.5, 2, 0.5);
      classLabels : array[0..1] of integer = (-1, 1);
begin
     fExamples.Free;
     FreeAndNil(fClassifier);
     FreeAndNil(fSVMBmp);

{$IFDEF INITRANDSEED}
     RandSeed := 100;
{$ENDIF}

     // testing a gaussian distribution
     fexamples := Create2DGaussSet(means, stdevs, classLabels, 2, 50);
     PaintBox1.Repaint;
end;

procedure TfrmClassifierTest.Button2Click(Sender: TObject);
var learner : TDecisionStumpLearner;
begin
     if Assigned(fExamples) then
     begin
          fClassifier.Free;
          learner := TDecisionStumpLearner.Create;
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

procedure TfrmClassifierTest.Button3Click(Sender: TObject);
var learner : TDiscreteAdaBoostLearner;
    props : TBoostProperties;
begin
     if Assigned(fExamples) then
     begin
          fClassifier.Free;

          learner := TDiscreteAdaBoostLearner.Create;
          try
             props.NumRounds := 100;
             props.PruneToLowestError := True;
             props.InitClassSpecificWeights := True;
             props.OwnsLearner := True;
             props.WeakLearner := TDecisionStumpLearner.Create;

             learner.SetProperties(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn;
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.Button4Click(Sender: TObject);
var learner : TGentleBoostLearner;
    props : TBoostProperties;
begin
     if Assigned(fExamples) then
     begin
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
             fClassifier := learner.Learn;
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.Button5Click(Sender: TObject);
var learner : TVotedBaggingLearner;
    props : TVotedBaggingProps;
begin
     if Assigned(fExamples) then
     begin
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
             fClassifier := learner.Learn;
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

procedure TfrmClassifierTest.Button7Click(Sender: TObject);
var props : TFisherAugmentedBaseProps;
    clProps : TFischerRobustLDAProps;
    examples : TCustomIncrementalLearnerExampleList;
begin
     if not DirectoryExists('.\Faces\') then
     begin
          ShowMessage('Face directory does not exists');
          exit;
     end;
     fExamples.Free;
     fClassifier.Free;

     props.UseFullSpace := False;
     if rbFisherLDA.Checked
     then
         props.ClassifierType := ctFast
     else if rbRobustFischer.Checked
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

     if Sender = Button7 then
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
             with TFischerIncrementalLDA.Create do
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

     if fClassifier is TFischerRobustLDAClassifier then
     begin
          clProps.NumHypothesis := 13;
          clProps.Start := 50;
          clProps.Stop := 20;
          clProps.ReductionFactor := 0.75;
          clProps.K2 := 0.01;
          clProps.accurFit := True;
          clProps.maxIter := 3;
          clProps.theta := 60;

          TFischerRobustLDAClassifier(fClassifier).SetProps(clProps);
     end;

     TestUnseenImages;

     PaintBox1.Repaint;
     TestLearnError;
end;

procedure TfrmClassifierTest.Button9Click(Sender: TObject);
var learner : TSVMLearner;
    props : TSVMProps;
begin
     if Assigned(fExamples) then
     begin
          fClassifier.Free;
          FreeAndNil(fSVMBmp);

          props.learnMethod := lmLagrangian;
          props.autoScale := True;
          props.kernelType := svmGauss;
          //props.order := 7;
          props.sigma := 0.51;
          props.slack := 1;

          learner := TSVMLearner.Create;
          try
             learner.SetProps(props);
             learner.Init(fExamples);
             fClassifier := learner.Learn;
          finally
                 learner.Free;
          end;
     end;

     PaintBox1.Repaint;
     TestLearnError;
end;

function TfrmClassifierTest.Create2DGaussSet(const mean,
  stddev: array of Double; const classLabels : Array of integer; numDim : integer; numExamples : integer): TCustomLearnerExampleList;
var i, j, k : integer;
    matrix : TDoubleMatrix;
    classvals : Array of integer;
    actClassIdx : integer;
begin
     assert(Length(mean) = length(stddev), 'Dimension error');
     assert(Length(mean) mod numDim = 0, 'Dimension error');
     assert(Length(classLabels) = Length(stddev) div numDim, 'Dimension error');
     Setlength(classvals, numExamples*(Length(stddev) div numDim));

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
          for j := 0 to numExamples - 1 do
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
     fSVMBmp.Free;

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

procedure PaintSVM(cl : TSVMClassifier);
var map : Array of Array of integer;
    aMtx : TDoubleMatrix;
    xmpl : TMatrixExample;
    x, y : integer;
    pts : Array of TPoint;
    numPts : integer;
const divisor : integer = 4;
begin
     if not Assigned(fSVMBmp) then
     begin
          fSVMBmp := TBitmap.Create;
          fSVMBmp.SetSize(PaintBox1.Width, PaintBox1.Height);

          // classify each pixel and set it's color if neighbouring pixels are different
          SetLength(map, PaintBox1.Height div divisor, PaintBox1.Width div divisor);

          aMtx := TDoubleMatrix.Create(1, 2);
          xmpl := TMatrixExample.Create(aMtx, false);
          for y := 0 to PaintBox1.Height div divisor - 1 do
          begin
               for x := 0 to PaintBox1.Width div divisor - 1 do
               begin
                    aMtx[0, 0] := minXVal + divisor*x/xDim;
                    aMtx[0, 1] := minYVal + divisor*y/yDim;
                    
                    map[PaintBox1.Height div divisor - y - 1][x] := cl.Classify(xmpl);
               end;
          end;
          xmpl.Free;
          aMtx.Free;

          fSVMBmp.Canvas.Brush.Color := PaintBox1.Color;
          fSVMBmp.Canvas.FillRect(Rect(0, 0, fSVMBmp.Width, fSVMBmp.Height));
          
          numPts := 0;
          // check in x direction
          SetLength(pts, (PaintBox1.Height div divisor)*(PaintBox1.Width div divisor));
          for y := 1 to Length(map) - 1 do
          begin
               for x := 1 to Length(map[0]) - 1 do
                   if ((map[y][x-1] <> map[y][x]) and (map[y][x-1] = -1)) or
                      ((map[y-1][x] <> map[y][x]) and (map[y-1][x] = -1)) then
                   begin
                        pts[numPts].X := x*divisor;
                        pts[numPts].Y := y*divisor;
                        inc(numPts);
                   end;
          end;

          fSVMBmp.Canvas.Polyline(Copy(pts, 0, numPts));
     end;

     PaintBox1.Canvas.StretchDraw(PaintBox1.ClientRect, fSVMBmp);
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

     if Assigned(fClassifier) and  (fClassifier is TSVMClassifier) then
        PaintSVM(TSVMClassifier(fClassifier)); 
     

     PaintBox1.Canvas.Brush.Style := bsSolid;
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

          if cl is TFischerLDAClassifier then
          begin
               PaintBox1.Canvas.Pen.Color := clMaroon;
               centers := TFischerLDAClassifier(cl).BackProjectedCenters;
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
end;

procedure TfrmClassifierTest.PaintFaceClassifier;
var faces : TDoubleMatrixDynArr;
begin
     // #################################################
     // #### Paint backprojected face class centers
     if (( (fClassifier is TFischerLDAClassifier) ) or ((fClassifier is TIncrementalFischerLDAClassifier))) and not Assigned(fFace1) then
     begin
          if (fClassifier is TFischerLDAClassifier) 
          then
              faces := (fClassifier as TFischerLDAClassifier).BackProjectedCenters
          else
              faces := (fClassifier as TIncrementalFischerLDAClassifier).BackProjectedCenters;

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
                   if fClassifier is TFischerLDAClassifier then
                      TFischerLDAClassifier(fClassifier).OnReconstructFace := OnPCAReconstruct;

                   if fClassifier is TIncrementalFischerLDAClassifier then
                      TIncrementalFischerLDAClassifier(fClassifier).OnReconstructFace := OnPCAReconstruct;
                   
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

        if fClassifier is TFischerLDAClassifier then
           TFischerLDAClassifier(fClassifier).OnReconstructFace := nil;
        lblUnseen.Caption := labels.CommaText;
     finally
            labels.Free;
     end;
end;

end.
