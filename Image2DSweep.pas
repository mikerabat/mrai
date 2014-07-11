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

unit Image2DSweep;

// ############################################################
// ##### class which can be used to sweep via a sliding window
// ##### over a matrix and classify the parts
// ############################################################

interface

uses BaseClassifier, matrix, contnrs;

type
  TClassRec = class(TObject)
    fx : integer;
    fy : integer;
    fwidth : integer;
    fheight : integer;
  private
    function GetBottom: integer;
    function GetRight: integer;
  public
    property x : integer read fx;
    property y : integer read fy;
    property Width : integer read fWidth;
    property Height : integer read fHeight;
    property Right : integer read GetRight;
    property Bottom : integer read GetBottom;

    constructor Create(ax, ay, aWidth, aHeight : integer);
  end;
  PClassRec = ^TClassRec;

type
  TClassRecList = class(TObjectList)
  private
    function GetItem(index: integer): TClassRec;
  public
    property Item[index : integer] : TClassRec read GetItem; default;

    constructor Create;
  end;

implementation

{ TClassRecList }

constructor TClassRecList.Create;
begin
     inherited Create(True);
end;

function TClassRecList.GetItem(index: integer): TClassRec;
begin
     Result := TClassRec(Items[index]);
end;

{ TClassRec }

constructor TClassRec.Create(ax, ay, aWidth, aHeight: integer);
begin
     fX := ax;
     fY := aY;
     fWidth := aWidth;
     fHeight := aHeight;
end;

function TClassRec.GetBottom: integer;
begin
     Result := fY + fHeight - 1;
end;

function TClassRec.GetRight: integer;
begin
     Result := fX + fWidth - 1;
end;

end.
