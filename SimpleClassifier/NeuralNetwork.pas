// ###################################################################
// #### This file is part of the artificial intelligence project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2016, Michael R. . All rights reserved.
// ####
// #### Unless required by applicable law or agreed to in writing, software
// #### distributed under the License is distributed on an "AS IS" BASIS,
// #### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// #### See the License for the specific language governing permissions and
// #### limitations under the License.
// ###################################################################

unit NeuralNetwork;

// ###########################################
// #### Artificial Feed Forward Neural Networks
// #### -> first version includes a very base
// #### backpropagation learning algorithm following the
// #### proposed algorithm from the neural network lecture of H. Bishof
// ###########################################

// -> the input neurons are normed suched that the maximum input
//    values are around the input examples mean and scaled.

interface

uses Types, BaseClassifier, BaseMathPersistence;

// base simple neuron
type
  TNeuronType = (ntLinear, ntExpSigmoid, ntTanSigmoid);
type
  TNeuron = class(TBaseMathPersistence)
  private
    fThresh : double;
    fBeta : double;
    fNeuralType : TNeuronType;
    fWeights : TDoubleDynArray;
  public
    function Feed(const Input : TDoubleDynArray) : double;
    function Derrive(const outputVal : double) : double;
    procedure RandomInit(const RangeMin : double = -1; const RangeMax : double = 1);

    procedure DefineProps; override;

    procedure OnLoadDoubleProperty(const Name : String; const Value : double); override;
    procedure OnLoadIntProperty(const Name : String; Value : integer); override;
    procedure OnLoadDoubleArr(const Name : String; const Value : TDoubleDynArray); override;

    constructor Create(NumInputs : integer; nnType : TNeuronType);
  end;

// one layer -> collection of neurons
type
  TNeuralLayer = class(TBaseMathPersistence)
  private
    fNeurons : Array of TNeuron;
    fType : TNeuronType;

    fLoadIdx : integer;
  public
    function Feed(const Input : TDoubleDynArray) : TDoubleDynArray;

    procedure DefineProps; override;
    procedure OnLoadBeginList(const Name : String; count : integer); override;
    function OnLoadObject(Obj : TBaseMathPersistence) : boolean; override;
    procedure OnLoadEndList; override;

    procedure OnLoadIntProperty(const Name : String; Value : integer); override;

    constructor Create(NumInputs, NumNeurons : integer; NeuronType : TNeuronType);
    destructor Destroy; override;
  end;

// a complete feed forward net
type
  TNeuralLayerRec = record
    NumNeurons : integer;
    NeuronType : TNeuronType;
  end;
  TNeuralLayerRecArr = Array of TNeuralLayerRec;
  TNeuralMinMax = packed Array[0..1] of double; //0..min, 1..max
  TFeedForwardNeuralNet = class(TBaseMathPersistence)
  private
    fLayer : Array of TNeuralLayer;
    fInputMinMax : TNeuralMinMax;
    fOutputMinMax : TNeuralMinMax;

    fLoadIdx : integer;
  public
    function Feed(const Input : TDoubleDynArray) : TDoubleDynArray;

    procedure DefineProps; override;
    procedure OnLoadBeginList(const Name : String; count : integer); override;
    function OnLoadObject(Obj : TBaseMathPersistence) : boolean; override;
    procedure OnLoadEndList; override;
    procedure OnLoadBinaryProperty(const Name : String; const Value; size : integer); override;

    constructor Create(NumInputs : integer; const layers : TNeuralLayerRecArr; const InputMinMax : TNeuralMinMax);
    destructor Destroy; override;
  end;


type
  TNeuralNetLearnAlg = (nnBackprop);
  TNeuralNetProps = record
    learnAlgorithm : TNeuralNetLearnAlg;
    layers : TNeuralLayerRecArr; // without the output layer!
    //InputMinMax : Array[0..1] of double; // maximum input values for all input neurons (perhaps do that for each neuron?)
    outputLayer : TNeuronType;   // output neuron type. (number of output neurons is the number of classes
    eta : double;                // learning rate
    maxNumIter : integer;        // maximum number of batch iterations
    minNumIter : integer;        // minimum number of batch iterations (despite the error change)
    stopOnMinDeltaErr : double;  // when training error change is lower than this delta then stop training
    numMinDeltaErr : integer;    // number of batch iterations that needs to be lower than stopOnMinDeltaErr
  end;

// ###########################################
// #### Feed forward neural net classifier
type
  TNeuralNet = class(TCustomClassifier)
  private
    fNet : TFeedForwardNeuralNet;
    fClasses : TIntegerDynArray;
  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; overload; override;

    function OnLoadObject(const Name : String; Obj : TBaseMathPersistence) : boolean; overload; override;
    procedure OnLoadIntArr(const Name : String; const Value : TIntegerDynArray); override;

    procedure DefineProps; override;

    constructor Create(aNet : TFeedForwardNeuralNet; aClassList : TIntegerDynArray);
    destructor Destroy; override;
  end;

// ###########################################
// #### Simple feed forward neural net learning class
type
  TNeuralNetLearner = class(TCustomLearner)
  private
    fProps : TNeuralNetProps;

    fOk : Array of TDoubleDynArray;
    fnumFeatures : integer;
    fdeltaK, fdeltaI : TDoubleDynArray;
    fnumCl : integer;
    fclassLabels : TIntegerDynArray;
    foutputExpAct : TDoubleDynArray;
    fMaxNumNeurons : integer;

    function DataSetMinMax : TNeuralMinMax;
    procedure UpdateWeights(deltaK : double; outputs : TDoubleDynArray; neuron : TNeuron);
    procedure BackProp(net : TFeedForwardNeuralNet; randSet: TCustomLearnerExampleList);
  protected
    function DoUnweightedLearn : TCustomClassifier; override;
  public
    procedure SetProps(const Props : TNeuralNetProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses Math, SysUtils;

// ################################################
// ##### persistence
// ################################################

const cClassLabels = 'labels';
      cNeuralNet = 'feedforwardnet';
      cNetLayers = 'layers';
      cInputMinMax = 'inputMinMax';
      cOutputMinMax = 'outputMinMax';

      cNeuronType = 'neuronType';
      cNeuronList = 'neurons';

      cNeuronThresh = 'neuronThresh';
      cNeuronWeights = 'neuronWeights';
      cNeuronBeta = 'neuronBeta';
{ TNeuron }

constructor TNeuron.Create(NumInputs: integer; nnType: TNeuronType);
begin
     inherited Create;

     fNeuralType := nnType;

     fBeta := 1;
     fThresh := 0;
     SetLength(fWeights, NumInputs);
end;

function TNeuron.Feed(const Input : TDoubleDynArray): double;
var i : integer;
begin
     assert(Length(input) = length(fWeights), 'Error input does not match learned weights');

     Result := 0;
     for i := 0 to Length(Input) - 1 do
         Result := Result + Input[i]*fWeights[i];

     case fNeuralType of
       //ntPerceptron: Result := ifthen(Result < fThresh, 0, 1);
       ntLinear: Result := Result - fThresh;
       ntExpSigmoid: Result := 1/(1 + exp(-fBeta*(Result - fThresh)));
       ntTanSigmoid: Result := tanh(fBeta*(Result - fThresh));
     end;
end;

procedure TNeuron.RandomInit(const RangeMin : double = -1; const RangeMax: double = 1);
var i : Integer;
begin
     for i := 0 to Length(fWeights) - 1 do
         fWeights[i] := Random*(rangeMax - rangeMin) + RangeMin;   
end;

function TNeuron.Derrive(const outputVal: double): double;
begin
     case fNeuralType of
       ntLinear: Result := fBeta;
       ntExpSigmoid: Result := fBeta*(outputVal*(1 - outputVal));
       ntTanSigmoid: Result := fBeta*(1 - sqr(outputVal));
     else
         Result := 0;
     end;
end;

procedure TNeuron.DefineProps;
begin
     inherited;

     AddDoubleProperty(cNeuronThresh, fThresh);
     AddDoubleProperty(cNeuronBeta, fBeta);
     AddIntProperty(cNeuronType, Integer(fNeuralType));
     AddDoubleArr(cNeuronWeights, fWeights);
end;

procedure TNeuron.OnLoadIntProperty(const Name: String; Value: integer);
begin
     if SameText(Name, cNeuronType)
     then
         fNeuralType := TNeuronType(Value)
     else
         inherited;
end;

procedure TNeuron.OnLoadDoubleProperty(const Name: String; const Value: double);
begin
     if SameText(Name, cNeuronThresh)
     then
         fThresh := Value
     else if SameText(Name, cNeuronBeta)
     then
         fBeta := Value
     else
         inherited;
end;

procedure TNeuron.OnLoadDoubleArr(const Name: String;
  const Value: TDoubleDynArray);
begin
     if SameText(Name, cNeuronWeights)
     then
         fWeights := Value
     else
         inherited;
end;

{ TNeuralLayer }

constructor TNeuralLayer.Create(NumInputs, NumNeurons: integer;
  NeuronType: TNeuronType);
var i : integer;
begin
     inherited Create;

     fType := NeuronType;
     SetLength(fNeurons, NumNeurons);

     for i := 0 to Length(fNeurons) - 1 do
     begin
          fNeurons[i] := TNeuron.Create(NumInputs, NeuronType);
          fNeurons[i].RandomInit(-1, 1);
     end;
end;

function TNeuralLayer.Feed(const Input: TDoubleDynArray): TDoubleDynArray;
var i : integer;
begin
     SetLength(Result, Length(fNeurons));

     for i := 0 to Length(fNeurons) - 1 do
         Result[i] := fNeurons[i].Feed(Input);
end;

destructor TNeuralLayer.Destroy;
var counter: Integer;
begin
     for counter := 0 to Length(fNeurons) - 1 do
         fNeurons[counter].Free;

     inherited;
end;

procedure TNeuralLayer.DefineProps;
var counter : integer;
begin
     inherited;

     AddIntProperty(cNeuronType, Integer(fType));
     BeginList(cNeuronList, Length(fNeurons));
     for counter := 0 to Length(fNeurons) - 1 do
         AddObject(fNeurons[counter]);
     EndList;
end;

procedure TNeuralLayer.OnLoadIntProperty(const Name: String; Value: integer);
begin
     if SameText(Name, cNeuronType)
     then
         fType := TNeuronType(Value)
     else
         inherited;
end;

procedure TNeuralLayer.OnLoadBeginList(const Name: String; count: integer);
begin
     fLoadIdx := -1;
     if SameText(Name, cNeuronList) then
     begin
          SetLength(fNeurons, count);
          fLoadIdx := 0;
     end
     else
         inherited;
end;

function TNeuralLayer.OnLoadObject(Obj: TBaseMathPersistence): boolean;
begin
     Result := True;
     if fLoadIdx >= 0 then
     begin
          fNeurons[fLoadIdx] := obj as TNeuron;
          inc(fLoadIdx);
     end
     else
         Result := inherited OnLoadObject(obj);
end;

procedure TNeuralLayer.OnLoadEndList;
begin
     if fLoadIdx >= 0
     then
         fLoadIdx := -1
     else
         inherited;
end;

{ TFeedForwardNeuralNet }

constructor TFeedForwardNeuralNet.Create(NumInputs: integer;
  const layers: TNeuralLayerRecArr; const InputMinMax : TNeuralMinMax);
var i : integer;
    lastNumInputs : integer;
procedure SetBetaAndThresh(aBeta, aThresh : double; layer : TNeuralLayer);
var counter : integer;
begin
     for counter := 0 to Length(layer.fNeurons) - 1 do
     begin
          layer.fNeurons[counter].fThresh := aThresh;
          layer.fNeurons[counter].fBeta := aBeta;
     end;
end;
begin
     SetLength(fLayer, Length(layers));

     fInputMinMax := InputMinMax;

     fOutputMinMax[1] := 1;
     case layers[High(layers)].NeuronType of
       ntLinear,
       ntTanSigmoid : fOutputMinMax[0] := -1;
       ntExpSigmoid : fOutputMinMax[0] := 0;
     end;

     lastNumInputs := NumInputs;
     for i := 0 to Length(fLayer) - 1 do
     begin
          fLayer[i] := TNeuralLayer.Create(lastNumInputs, layers[i].NumNeurons, layers[i].NeuronType);
          lastNumInputs := layers[i].NumNeurons;
     end;

     // adjust the input activation by the min max values and according to the input neuron type
     case fLayer[0].fType of
       ntLinear:  SetBetaAndThresh( 1, (fInputMinMax[0] + fInputMinMax[1])/2, fLayer[0]);
       ntExpSigmoid: SetBetaAndThresh(10/(fInputMinMax[1] - fInputMinMax[0]), (fInputMinMax[0] + fInputMinMax[1])/2, fLayer[0]);
       ntTanSigmoid: SetBetaAndThresh(2*pi/(fInputMinMax[1] - fInputMinMax[0]), (fInputMinMax[0] + fInputMinMax[1])/2, fLayer[0]);
     end;

     for i := 1 to Length(fLayer) - 1 do
     begin
          if fLayer[i].fType = ntLinear then
          begin
               if fLayer[i - 1].fType = ntExpSigmoid then
                  SetBetaAndThresh(1, 0.5, fLayer[i]);
          end
          else if fLayer[i].fType = ntTanSigmoid then
          begin
               if fLayer[i].fType = ntExpSigmoid then
                  SetBetaAndThresh(1, 0.5, fLayer[i]);
          end;
     end;

end;

destructor TFeedForwardNeuralNet.Destroy;
var i : integer;
begin
     for i := 0 to Length(fLayer) - 1 do
         fLayer[i].Free;

     inherited;
end;

function TFeedForwardNeuralNet.Feed(const Input: TDoubleDynArray): TDoubleDynArray;
var i : integer;
begin
     SetLength(Result, Length(input));
     Result := Input;

     for i := 0 to Length(fLayer) - 1 do
         Result := fLayer[i].Feed(Result);
end;

procedure TFeedForwardNeuralNet.DefineProps;
var counter: Integer;
begin
     inherited;

     BeginList(cNetLayers, Length(fLayer));
     for counter := 0 to Length(fLayer) - 1 do
         AddObject(fLayer[counter]);
     EndList;

     AddBinaryProperty(cInputMinMax, fInputMinMax, sizeof(fInputMinMax));
     AddBinaryProperty(cOutputMinMax, fOutputMinMax, sizeof(fOutputMinMax));
end;

procedure TFeedForwardNeuralNet.OnLoadBinaryProperty(const Name: String;
  const Value; size: integer);
begin
     if SameText(Name, cInputMinMax) then
     begin
          assert(Size = sizeof(fInputMinMax), 'Error persistent size of Input Min Max differs');
          Move(Value, fInputMinMax, Size);
     end
     else if SameText(Name, cOutputMinMax) then
     begin
          assert(Size = sizeof(fInputMinMax), 'Error persistent size of Input Min Max differs');
          Move(Value, fOutputMinMax, Size);
     end
     else
         inherited;
end;

procedure TFeedForwardNeuralNet.OnLoadBeginList(const Name: String;
  count: integer);
begin
     fLoadIdx := -1;
     if SameText(Name, cNetLayers) then
     begin
          SetLength(fLayer, count);
          fLoadIdx := 0;
     end
     else
         inherited;
end;

procedure TFeedForwardNeuralNet.OnLoadEndList;
begin
     if fLoadIdx <> -1
     then
         fLoadIdx := -1
     else
         inherited;
end;

function TFeedForwardNeuralNet.OnLoadObject(Obj: TBaseMathPersistence): boolean;
begin
     Result := True;
     if fLoadIdx >= 0 then
     begin
          fLayer[fLoadIdx] := obj as TNeuralLayer;
          inc(fLoadIdx);
     end
     else
         Result := inherited OnLoadObject(Obj);
end;

{ TNeuralNetLearner }

procedure TNeuralNetLearner.BackProp(net : TFeedForwardNeuralNet; randSet : TCustomLearnerExampleList);
var exmplCnt : integer;
    counter : integer;
    actIdx : integer;
    layerCnt : integer;
    lastClass : integer;
    neuronCnt : integer;
    tmp : TDoubleDynArray;
begin
     // init section
     if fOk = nil then
     begin
          SetLength(fOk, 2 + Length(fProps.layers));
          SetLength(fOk[0], fnumFeatures);
          SetLength(foutputExpAct, fnumCl);

          SetLength(fdeltak, fMaxNumNeurons);
          SetLength(fdeltaI, fMaxNumNeurons);
     end;

     lastClass := MaxInt;

     // ###############################################
     // #### Performs one batch backpropagations step on the given randomized data set
     // it reuses some global object variables
     for exmplCnt := 0 to randSet.Count - 1 do
     begin
          // define wanted output activation for the current example
          if randSet.Example[exmplCnt].ClassVal <> lastClass then
          begin
               actIdx := -1;
               for counter := 0 to fNumCl - 1 do
               begin
                    if fClassLabels[counter] = randSet.Example[exmplCnt].ClassVal then
                    begin
                         actIdx := counter;
                         break;
                    end;
               end;

               for counter := 0 to fnumCl - 1 do
                   foutputExpAct[counter] := net.fOutputMinMax[IfThen(counter = actIdx, 1, 0) ];
          end;

          for counter := 0 to fNumFeatures - 1 do
              fOk[0][counter] := randSet[exmplCnt].FeatureVec[counter];

          // calculate activation (o_k) for all layers
          for layerCnt := 0 to Length(net.fLayer) - 1 do
              fOk[layerCnt + 1] := net.fLayer[layerCnt].Feed(fOk[layerCnt]);

          // output layer -> activation error is calculatead against the expected output
          for neuronCnt := 0 to Length(fok[length(fok) - 1]) - 1 do
          begin
               fdeltak[neuronCnt] := (foutputExpAct[neuronCnt] - fok[Length(fok) - 1][neuronCnt])*
                                     net.fLayer[Length(net.fLayer) - 1].fNeurons[neuronCnt].Derrive(fok[Length(fok) - 1][neuronCnt]);

               // update weights of the final neuron
               UpdateWeights(fdeltaK[neuronCnt], fok[Length(fok) - 2], net.fLayer[Length(net.fLayer) - 1].fNeurons[neuronCnt]);
          end;

          // propagate error through the layers and update weights
          for layerCnt := Length(fProps.layers) - 1 downto 0 do
          begin
               // switch deltai and deltak (faster reallocation ;) )
               tmp := fdeltaK;
               fdeltaK := fdeltaI;
               fdeltaI := tmp;

               // #############################################
               // #### backpropagation step
               for neuronCnt := 0 to fProps.layers[layerCnt].NumNeurons - 1 do
               begin
                    fdeltaK[neuronCnt] := 0;

                    for counter := 0 to Length(net.fLayer[layerCnt + 1].fNeurons) - 1 do
                        fdeltaK[neuronCnt] := fdeltaK[neuronCnt] +
                                              fdeltaI[counter]*
                                              net.fLayer[layerCnt + 1].fNeurons[counter].fWeights[neuronCnt];
                    fdeltaK[neuronCnt] := fdeltaK[neuronCnt]*net.fLayer[layerCnt].fNeurons[neuronCnt].Derrive(fOk[layerCnt + 1][neuronCnt]);

                    // update weights of the current neuron
                    UpdateWeights(fdeltaK[neuronCnt], fOk[layerCnt], net.fLayer[layercnt].fNeurons[neuronCnt]);
               end;
          end;
     end;
end;

function TNeuralNetLearner.DoUnweightedLearn: TCustomClassifier;
var net : TFeedForwardNeuralNet;
    layers : TNeuralLayerRecArr;
    counter: Integer;
    learnCnt : integer;
    lastErr, curErr : double;
    numSmallErrChange : integer;
    errCnt : integer;
    inputMinMax : TNeuralMinMax;
    randSet : TCustomLearnerExampleList;
begin
     fclassLabels := Classes;
     fnumCl := Length(fclassLabels);
     fnumFeatures := DataSet.Example[0].FeatureVec.FeatureVecLen;

     fMaxNumNeurons := Max(fNumCl, fnumFeatures);

     for counter := 0 to Length(fProps.layers) - 1 do
         fMaxNumNeurons := Max(fMaxNumNeurons, fProps.layers[counter].NumNeurons);

     SetLength(layers, Length(fProps.layers) + 1);
     layers[Length(layers) - 1].NeuronType := fProps.outputLayer;
     layers[Length(layers) - 1].NumNeurons := fnumCl;

     inputMinMax := DataSetMinMax;

     if Length(fProps.layers) > 0 then
        Move(fProps.layers[0], layers[0], Length(fProps.layers)*sizeof(fProps.layers[0]));

     net := TFeedForwardNeuralNet.Create(fnumFeatures, layers, inputMinMax);

     lastErr := 1;
     numSmallErrChange := 0;

     // ###########################################
     // #### create classifier
     Result := TNeuralNet.Create(net, Classes);

     // #########################################################
     // ##### batch error evaluation
     for learnCnt := 0 to fProps.maxNumIter - 1 do
     begin

          // ##########################################################
          // #### One batch learn iteration (randomized data set)
          randSet := DataSet.CreateRandomDataSet(100);
          try
             Backprop(net, randSet);
          finally
                 randSet.Free;
          end;


          // ##################################################
          // #### test learning error
          curErr := 0;
          for errCnt := 0 to DataSet.Count - 1 do
          begin
               if Result.Classify(DataSet.Example[errCnt]) <> DataSet.Example[errCnt].ClassVal then
                  curErr := curErr + 1;
          end;
          curErr := curErr/DataSet.Count;

          if (learnCnt >= fProps.minNumIter) and (lastErr - curErr < fProps.stopOnMinDeltaErr) then
          begin
               inc(numSmallErrChange);

               if numSmallErrChange >= fProps.numMinDeltaErr then
                  break;
          end
          else
          begin
               lastErr := curErr;
               numSmallErrChange := 0;
          end;
     end;
end;

procedure TNeuralNetLearner.UpdateWeights(deltaK: double;
  outputs: TDoubleDynArray; neuron: TNeuron);
var counter : integer;
begin
     // very simple update rule (simple gradient decent with learning factor eta)
     for counter := 0 to Length(neuron.fWeights) - 1 do
         neuron.fWeights[counter] := neuron.fWeights[counter] + fProps.eta*deltaK*outputs[counter];
end;

procedure TNeuralNetLearner.SetProps(const Props: TNeuralNetProps);
begin
     fProps := Props;
end;

function TNeuralNetLearner.DataSetMinMax: TNeuralMinMax;
var counter : integer;
    featureCnt : integer;
begin
     Result[0] := DataSet[0].FeatureVec[0];
     Result[1] := DataSet[0].FeatureVec[0];

     for counter := 0 to DataSet.Count - 1 do
     begin
          for featureCnt := 0 to DataSet[0].FeatureVec.FeatureVecLen - 1 do
          begin
               Result[0] := Min(Result[0], DataSet[counter].FeatureVec[featureCnt]);
               Result[1] := Max(Result[1], DataSet[counter].FeatureVec[featureCnt]);
          end;
     end;
end;

class function TNeuralNetLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TNeuralNet;
end;

{ TNeuralNet }

function TNeuralNet.Classify(Example: TCustomExample;
  var confidence: double): integer;
var ok : TDoubleDynArray;
    layerCnt : integer;
    counter: Integer;
    maxIdx : integer;
begin
     confidence := 0;

     SetLength(ok, Example.FeatureVec.FeatureVecLen);
     // ###########################################
     // #### Restrict input to defined min max
     for counter := 0 to Example.FeatureVec.FeatureVecLen - 1 do
         ok[counter] := Max(Min(Example.FeatureVec[counter], fNet.fInputMinMax[1]), fNet.fInputMinMax[0]);

     for layerCnt := 0 to Length(fNet.fLayer) - 1 do
         ok := fNet.fLayer[layerCnt].Feed(ok);
      
     // maximum wins
     maxIdx := 0;

     for counter := 0 to Length(ok) - 1 do
         if ok[maxIdx] < ok[counter] then
            maxIdx := counter;

     Result := fClasses[maxIdx];
end;

constructor TNeuralNet.Create(aNet: TFeedForwardNeuralNet; aClassList : TIntegerDynArray);
begin
     fNet := aNet;
     fClasses := aClassList;

     inherited Create;
end;

destructor TNeuralNet.Destroy;
begin
     fNet.Free;

     inherited;
end;

function TNeuralNet.OnLoadObject(const Name: String;
  Obj: TBaseMathPersistence): boolean;
begin
     Result := True;
     if SameText(Name, cNeuralNet)
     then
         fNet := Obj as TFeedForwardNeuralNet
     else
         Result := inherited OnLoadObject(Name, obj);
end;

procedure TNeuralNet.OnLoadIntArr(const Name: String;
  const Value: TIntegerDynArray);
begin
     if SameText(Name, cClassLabels)
     then
         fClasses := Value
     else
         inherited;
end;

procedure TNeuralNet.DefineProps;
begin
     inherited;

     AddIntArr(cClassLabels, fClasses);
     AddObject(cNeuralNet, fNet);
end;

initialization
  RegisterMathIO(TNeuralNet);
  RegisterMathIO(TFeedForwardNeuralNet);
  RegisterMathIO(TNeuralLayer);
  RegisterMathIO(TNeuron);


end.
