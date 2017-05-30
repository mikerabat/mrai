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

unit TreeStructs;

// #############################################################
// #### Tree structures used in the tree based classifiers
// #############################################################

interface

uses SysUtils, Classes;

type
  TCustomTreeItem = class(TObject)
  private
    fTreeData: TObject;
    fParent : TCustomTreeItem;
    procedure SetTreeData(const Data : TObject);
  public
    property TreeData : TObject read fTreeData write SetTreeData;
    property Parent : TCustomTreeItem read fParent;

    destructor Destroy; override;
  end;

type
  TTreeNode = class(TCustomTreeItem)
  private
    fLeftItem: TCustomTreeItem;
    fRightItem: TCustomTreeItem;
    procedure SetLeftItem(item : TCustomTreeItem);
    procedure SetRightItem(item : TCustomTreeItem);
  public
    property LeftItem : TCustomTreeItem read fLeftItem write SetLeftItem;
    property RightItem : TCustomTreeItem read fRightItem write SetRightItem;

    destructor Destroy; override;
  end;

type
  TTreeLeave = class(TCustomTreeItem)
  end;

implementation

{ TCustomTreeItem }

destructor TCustomTreeItem.Destroy;
begin
     FreeAndNil(fTreeData);

     inherited;
end;

procedure TCustomTreeItem.SetTreeData(const Data: TObject);
begin
     if Assigned(fTreeData) then
        FreeAndNil(fTreeData);

     fTreeData := Data;
end;

{ TTreeNode }

destructor TTreeNode.Destroy;
begin
     // in the final tree state we own the "tree" so free the items
     fLeftItem.Free;
     fRightItem.Free;

     inherited;
end;

procedure TTreeNode.SetLeftItem(item: TCustomTreeItem);
begin
     // for external operations -> do not free the items
     fLeftItem := item;
     item.fParent := self;
end;

procedure TTreeNode.SetRightItem(item: TCustomTreeItem);
begin
     fRightItem := item;
     item.fParent := self;
end;

end.
