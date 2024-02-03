class ReidDataset(Dataset):
    
    def __init__(self, query_path, transform = None):
        super().__init__()
        self.query_path = query_path
        assert os.path.exists(self.query_path) and os.path.isdir(self.query_path), "Given path '{}' is not exists or directory".format(self.query_path)
    
        self.path_list = os.listdir(query_path)
        self.transform = transform
        
    def __len__(self):
        
        return len(self.path_list)
        
    def __getitem__(self, idx):
      
        file_path = os.path.join(self.query_path, self.path_list[idx])
        print(self.path_list[idx])
        img = read_image(file_path)
        img = img.float()
      
        if self.transform is not None:
            img = self.transform(img)
    
        return label, img
            
