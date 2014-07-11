unit NeuralNetwork;

// ###########################################
// #### Artificial Feed Forward Neural Networks
// ###########################################

interface

uses Types;

// base simple neuron
type
  TNeuronType = (ntPerceptron, ntLinear, ntExpSigmoid, ntTanSigmoid);
type
  TNeuron = class(TObject)
  private
    fThresh : double;
    fBeta : double;
    fNeuralType : TNeuronType;
    fWeights : TDoubleDynArray;
  public
    function Feed(const Input : TDoubleDynArray) : double;
    procedure RandomInit(const RangeMin : double = -1; const RangeMax : double = 1);
    
    constructor CreateNew(NumInputs : integer; nnType : TNeuronType);
  end;

// one layer -> collection of neurons
type
  TNeuralLayer = class(TObject)
  private
    fNeurons : Array of TNeuron;
    fType : TNeuronType;
  public
    function Feed(const Input : TDoubleDynArray) : TDoubleDynArray;
    
    constructor CreateNew(NumInputs, NumNeurons : integer; NeuronType : TNeuronType);
  end;

// a complete feed forward net
type
  TFeedForwardNeuralNet = class(TObject)
  private
    fLayer : Array of TNeuralLayer;  
    fInputMinMax : Array[0..1] of TDoubleDynArray;
    fOuptutMinMax : Array[0..1] of TDoubleDynArray;
  public
    constructor CreateNew(NumInputs : integer; const NeuronsInLayer : Array of integer; const NeuronTypes : Array of TNeuronType);

    function Feed(const Input : TDoubleDynArray) : TDoubleDynArray;
    procedure BackProp(const Input, Output : TDoubleDynArray);

    destructor Destroy; override;
  end;

implementation

uses Math;

{ TNeuron }

constructor TNeuron.CreateNew(NumInputs: integer; nnType: TNeuronType);
begin
     fNeuralType := nnType;

     SetLength(fWeights, NumInputs);
     RandomInit(0, 0);
end;

function TNeuron.Feed(const Input: array of double): double;
var i : integer;
begin
     assert(Length(input) = length(fWeights), 'Error input does not match learned weights');

     Result := 0;
     for i := 0 to Length(Input) - 1 do
         Result := Result + Input[i]*fWeights[i];

     case fNeuralType of
       ntPerceptron: Result := ifthen(Result < fThresh, 0, 1);
       ntLinear: Result := Result + fThresh;
       ntExpSigmoid: Result := 1/(1 + exp(-fBeta*(Result + fThresh)));
       ntTanSigmoid: Result := tanh(Result + fThresh);
     end;
end;

procedure TNeuron.RandomInit(const RangeMin : double = -1; const RangeMax: double = 1);
var i : Integer;
begin
     for i := 0 to Length(fWeights) - 1 do
         fWeights[i] := Random*(rangeMax - rangeMin) + RangeMin;   
end;

{ TNeuralLayer }

constructor TNeuralLayer.CreateNew(NumInputs, NumNeurons: integer;
  NeuronType: TNeuronType);
var i : integer;
begin
     fType := NeuronType;
     SetLength(fNeurons, NumNeurons);

     for i := 0 to Length(fNeurons) - 1 do
     begin
          fNeurons[i] := TNeuron.CreateNew(NumInputs, NeuronType);
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

{ TFeedForwardNeuralNet }

constructor TFeedForwardNeuralNet.CreateNew(NumInputs : integer;
  const NeuronsInLayer: array of integer; const NeuronTypes : Array of TNeuronType);
var i : integer;
    lastNumInputs : integer;
begin
     assert(Length(NeuronsInLayer) = Length(NeuronTypes), 'Error neuron arrays need the same length');
     SetLength(fLayer, Length(NeuronsInLayer));
     SetLength(fInputMinMax[0], NumInputs);
     SetLength(fInputMinMax[1], NumInputs);
     SetLength(fOuptutMinMax[0], NeuronsInLayer[High(NeuronsInLayer)]);
     SetLength(fOuptutMinMax[1], NeuronsInLayer[High(NeuronsInLayer)]);

     for i := 0 to Length(fInputMinMax[0]) - 1 do
     begin
          fInputMinMax[0][i] := -1;
          fInputMinMax[1][i] := +1;
     end;
     
     lastNumInputs := NumInputs;
     for i := 0 to Length(fLayer) - 1 do
     begin
          fLayer[i] :=  TNeuralLayer.CreateNew(lastNumInputs, NeuronsInLayer[i], NeuronTypes[i]);
          lastNumInputs := NeuronsInLayer[i];
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

end.
