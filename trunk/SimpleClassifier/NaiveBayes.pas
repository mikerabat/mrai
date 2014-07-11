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

unit NaiveBayes;

// ######################################################
// #### Naive Bayes Classifier
// ######################################################

interface

uses SysUtils, Classes, Types, BaseClassifier;

type
  TNaiveBayesProps = record
    HistoMin : double;
    HistoMax : double;
    NumBins : integer;
  end;

type
  TDoubleDynArrayArr = Array of TDoubleDynArray;

// ######################################################
// #### Classifies the example according to a naive
// #### Bayes approach.
// classification is argmax_C(P(x1|C)*P(x2|C)*...*P(xn|C))
type
  TNaiveBayes = class(TCustomClassifier)
  private
    fProps : TNaiveBayesProps;
    fDistr : TDoubleDynArrayArr;
    fClVals : TIntegerDynArray;
    fIdx : integer;

  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; overload; override;
    function Classify(Example : TCustomExample) : integer; overload; override;

    // loading and saving procedures
    procedure OnLoadDoubleProperty(const Name : String; const Value : double); override;
    procedure OnLoadIntProperty(const Name : String; Value : integer); override;
    procedure OnLoadIntArr(const Name : String; const Value : TIntegerDynArray); override;
    procedure OnLoadListDoubleArr(const Value: TDoubleDynArray); override;
    procedure OnLoadBeginList(const Name : String; count : integer); override;
    procedure OnLoadEndList; override;

    procedure DefineProps; override;
    constructor Create(const props : TNaiveBayesProps; const distr : TDoubleDynArrayArr; const clVals : TIntegerDynArray);
  end;

// #####################################################
// #### Naive Bayes learner. The classifier is based
// on histograms with a specific bin size and assumes
// independent features.
type
  TNaiveBayesLearner = class(TCustomWeightedLearner)
  private
    fProps : TNaiveBayesProps;

  protected
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    procedure AfterConstruction; override;

    procedure SetProps(const Props : TNaiveBayesProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses Math, BaseMathPersistence;

// ################################################
// ##### persistence
// ################################################

const cClassLabels = 'labels';
      cDistrList = 'distrList';
      cDistribution = 'distr';
      cNumBins = 'numbins';
      cHistoMin = 'histmin';
      cHistoMax = 'histmax';


{ TNaiveBayesLearner }

procedure TNaiveBayesLearner.AfterConstruction;
begin
     inherited;

     fProps.HistoMin := 0;
     fProps.HistoMax := 1;
     fProps.NumBins := 100;
end;

class function TNaiveBayesLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := classifier = TNaiveBayes;
end;

function TNaiveBayesLearner.DoLearn(
  const weights: array of double): TCustomClassifier;
var numClasses : integer;
    cl : TIntegerDynArray;
    clIdx : integer;
    counter : integer;
    distr : TDoubleDynArrayArr;
    histIdx : integer;
    clCnt : integer;
    featureIdx: Integer;
    featureVal : double;
    example : TCustomLearnerExample;
    distrCnt : Array of TDoubleDynArray;
    histoDiff : double;
    maxHistVal : double;
begin
     numClasses := DataSet.NumClasses(cl);

     if numClasses = 0 then
        raise Exception.Create('Error no data.');

     histoDiff := fProps.HistoMax - fProps.HistoMin;
     maxHistVal := fProps.HistoMax - histoDiff/(100*fProps.NumBins); // features at the edge may result in range check errors!


     // create a histogram for each feature and class
     SetLength(distr, numClasses);
     SetLength(distrCnt, numClasses);
     for clIdx := 0 to numClasses - 1 do
     begin
          SetLength(distr[clIdx], DataSet.Example[0].FeatureVec.FeatureVecLen*fProps.NumBins);
          SetLength(distrCnt[clIdx], DataSet.Example[0].FeatureVec.FeatureVecLen);
     end;

     for counter := 0 to DataSet.Count - 1 do
     begin
          example := DataSet.Example[counter];
          for featureIdx := 0 to example.FeatureVec.FeatureVecLen - 1 do
          begin
               // truncate the feature value to the expected min max!
               featureVal := Min(maxHistVal, Max(fProps.HistoMin, example.FeatureVec[featureIdx]));

               // find class
               for clCnt := 0 to numClasses - 1 do
               begin
                    if cl[clCnt] = example.ClassVal then
                    begin
                         histIdx := fProps.NumBins*featureIdx + Floor((featureVal - fProps.HistoMin)/histoDiff*fProps.NumBins);

                         // do a weighted add
                         distrCnt[clCnt][featureIdx] := distrCnt[clCnt][featureIdx] + weights[counter];
                         distr[clCnt][histIdx] := distr[clCnt][histIdx] + weights[counter];
                    end;
               end;
          end;
     end;

     // create the distribution
     for clCnt := 0 to numClasses - 1 do
     begin
          for featureIdx := 0 to DataSet.Example[0].FeatureVec.FeatureVecLen - 1 do
          begin
               for histIdx := 0 to fProps.NumBins - 1 do
               begin
                    if distrCnt[clCnt][featureIdx] <> 0 then
                       distr[clCnt][histIdx + featureIdx*fProps.NumBins] := distr[clCnt][histIdx + featureIdx*fProps.NumBins]/distrCnt[clCnt][featureIdx];
               end;
          end;
     end;


     // #############################################################
     // #### Create the classifier
     Result := TNaiveBayes.Create(fProps, distr, cl);
end;

procedure TNaiveBayesLearner.SetProps(const Props: TNaiveBayesProps);
begin
     fProps := props;
end;


// ####################################################
// #### Naive Bayes classifier
// ####################################################

function TNaiveBayes.Classify(Example: TCustomExample;
  var confidence: double): integer;
var clCnt: Integer;
    prob : TDoubleDynArray;
    featureIdx: Integer;
    feature : double;
    histoIdx : integer;
    histoDiff : double;
    maxHistVal : double;
begin
     confidence := 0;

     SetLength(prob, Length(fClVals));
     for clCnt := 0 to Length(prob) - 1 do
         prob[clCnt] := 1;

     histoDiff := fProps.HistoMax - fProps.HistoMin;
     maxHistVal := fProps.HistoMax - histoDiff/(100*fProps.NumBins); // features at the edge may result in range check errors!

     // search for the winning (most probable) class
     // and handle each feature independently meaning that
     // the result is the product of all features!

     // note we want to evalute the features only once!
     for featureIdx := 0 to Example.FeatureVec.FeatureVecLen - 1 do
     begin
          feature := Max(fProps.HistoMin, Min(maxHistVal, Example.FeatureVec[featureIdx]));
          histoIdx := featureIdx*fProps.NumBins + Floor((feature - fProps.HistoMin)/histoDiff*fProps.NumBins);
          for clCnt := 0 to Length(fClVals) - 1 do
              prob[clCnt] := prob[clCnt]*fDistr[clCnt][histoIdx];
     end;

     // find maximum
     confidence := prob[0];
     Result := fClVals[0];

     for clCnt := 1 to Length(fClVals) - 1 do
     begin
          if confidence < prob[clCnt] then
          begin
               Result := fClVals[clCnt];
               confidence := prob[clCnt];
          end;
     end;
end;

function TNaiveBayes.Classify(Example: TCustomExample): integer;
var conf : double;
begin
     Result := Classify(Example, conf);
end;

constructor TNaiveBayes.Create(const props: TNaiveBayesProps;
  const distr: TDoubleDynArrayArr; const clVals: TIntegerDynArray);
begin
     inherited Create;

     fProps := props;
     fDistr := distr;
     fClVals := clVals;
end;

procedure TNaiveBayes.DefineProps;
var
  counter: Integer;
begin
     inherited;

     AddIntArr(cClassLabels, fClVals);
     AddIntProperty(cNumBins, fProps.NumBins);
     AddDoubleProperty(cHistoMin, fProps.HistoMin);
     AddDoubleProperty(cHistoMax, fProps.HistoMax);

     BeginList(cDistrList, Length(fDistr));
     for counter := 0 to Length(fDistr) - 1 do
         AddDoubleArr(cDistribution, fDistr[counter]);
     EndList;
end;

procedure TNaiveBayes.OnLoadBeginList(const Name: String; count: integer);
begin
     fIdx := -1;

     if SameText(Name, cDistrList) then
     begin
          SetLength(fDistr, count);
          fIdx := 0;
     end
     else
         inherited;
end;

procedure TNaiveBayes.OnLoadDoubleProperty(const Name: String;
  const Value: double);
begin
     if SameText(Name, cHistoMin)
     then
         fProps.HistoMin := Value
     else if SameText(Name, cHistoMax)
     then
         fProps.HistoMax := Value
     else
         inherited;
end;

procedure TNaiveBayes.OnLoadListDoubleArr(const Value : TDoubleDynArray);
begin
     if fIdx >= 0 then
     begin
          fDistr[fIdx] := Value;
          inc(fIdx);
     end
     else
         inherited;
end;

procedure TNaiveBayes.OnLoadEndList;
begin
     inherited;
end;

procedure TNaiveBayes.OnLoadIntArr(const Name: String;
  const Value: TIntegerDynArray);
begin
     if SameText(Name, cClassLabels)
     then
         fClVals := Value
     else
         inherited;
end;

procedure TNaiveBayes.OnLoadIntProperty(const Name: String; Value: integer);
begin
     if SameText(Name, cNumBins)
     then
         fProps.NumBins := Value
     else
         inherited;
end;

initialization
  RegisterMathIO(TNaiveBayes);

end.
