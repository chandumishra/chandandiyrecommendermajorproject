// SPDX-License-Identifier: MIT 
pragma solidity ^0.6.0; 
pragma experimental ABIEncoderV2; 
contract IdeaRecommendation { 
struct DiyIdea { 
string ideaName; 
} 
struct Instance { 
string IID; 
string ImageData; 
uint256 DiyListSize; 
bool Validity; 
} 
mapping(string => uint256) private diyListSizes; 
mapping(string => mapping(uint256 => DiyIdea)) private diyIdeas; 
mapping(string => Instance) private instances; 
event Received(address indexed sender, uint amount); 
event VerifyResult(bool validity); 
function createAndInsertInstance(string memory IID, string memory imageData, uint256 diyListSize, 
string[] memory ideas) public { 
require(!isDiyRecordExist(IID), "This instance record already exists."); 
uint256 size = diyListSizes[IID]; 
for (uint256 i = 0; i < diyListSize; i++) { 
diyIdeas[IID][size + i] = DiyIdea(ideas[i]); 
} 
Instance memory input; 
input.IID = IID; 
input.ImageData = imageData; 
input.DiyListSize = diyListSize; 
input.Validity = true; 
//require(!isDiyRecordExist(IID), "This instance record already exists."); 
instances[IID] = input; 
diyListSizes[IID] += diyListSize; 
} 
function isDiyRecordExist(string memory IID) internal view returns (bool) { 
return instances[IID].Validity; 
} 
function verifyRecommendedIdeas(string memory iid) public returns (bool) { 
bool result = instances[iid].Validity; 
emit VerifyResult(result); 
return result; 
} 
function invalidateInstanceRecord(string memory iid) public { 
if (instances[iid].Validity) { 
instances[iid].Validity = false; 
} 
} 
fallback() external payable { 
emit Received(msg.sender, msg.value); 
// Additional logic can be added here when the contract receives Ether. 
// This function will be called if Ether is sent to the contract without any data. 
} 
receive() external payable { 
emit Received(msg.sender, msg.value); 
// Additional logic can be added here when the contract receives Ether. 
// This function will be called if Ether is sent to the contract without any data. 
} 
}
